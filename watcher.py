import json, time, yaml, os, sys
from datetime import datetime
import numpy as np
import cv2
from mss import mss
from PIL import Image
import pytesseract
from pync import Notifier

CONFIG_PATH = "config.json"
RULES_PATH  = "rules.yaml"
TEMPLATES_DIR = "templates"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(bbox):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"bbox": bbox}, f)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)["bbox"]

def calibrate_bounding_box():
    """
    Modo simples de calibração:
    1) maximize a janela do scrcpy no lugar/tamanho desejado
    2) rode: python watcher.py calibrate
    3) o script vai tirar uma screenshot pra te guiar
    4) você digita manualmente os 4 números do retângulo: left, top, width, height
       (dica: no mac, cmd+shift+4 mostra coords; ou use um utilitário de coordenadas)
    """
    print("\n=== Calibração da área do scrcpy ===")
    print("Dica: use cmd+shift+4 para ver coordenadas. Informe:")
    left = int(input("left: ").strip())
    top = int(input("top: ").strip())
    width = int(input("width: ").strip())
    height = int(input("height: ").strip())
    save_config({"left": left, "top": top, "width": width, "height": height})
    print(f"Salvo em {CONFIG_PATH}: {left,top,width,height}")

def grab_frame(sct, bbox):
    frame = np.array(sct.grab(bbox))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def find_yellow_cards(img_bgr):
    """Detecta os dois cards amarelos por cor (HSV) + contornos retangulares grandes."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # faixa ampla de "amarelo quente" (ajuste fino conforme seu monitor/tema)
    lower = np.array([15, 60, 120], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # limpa ruído
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    h, w = img_bgr.shape[:2]
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # heurísticas: área grande e proporção ~ retângulo card
        if area > (w*h)*0.03 and 0.8 < (cw / max(ch,1)) < 2.0:
            cards.append((x, y, cw, ch))
    # ordenar da esquerda pra direita
    cards = sorted(cards, key=lambda r: r[0])
    return cards

def ocr_text(img_bgr):
    # Preprocess simples p/ OCR
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cfg = "--psm 6 -l eng"
    data = pytesseract.image_to_data(th, output_type=pytesseract.Output.DICT, config=cfg)
    # Junta tudo; confianza média simples
    text = " ".join([t for t in data["text"] if t.strip()])
    conf = np.mean([float(c) for c in data["conf"] if c.isdigit() and int(c) >= 0]) if data["conf"] else 0.0
    return text, (conf/100.0 if conf else 0.0)

def load_templates():
    tmpls = {}
    for name in os.listdir(TEMPLATES_DIR):
        if name.lower().endswith((".png",".jpg",".jpeg")):
            key = os.path.splitext(name)[0]
            img = cv2.imread(os.path.join(TEMPLATES_DIR, name), cv2.IMREAD_UNCHANGED)
            if img is not None:
                tmpls[key] = img
    return tmpls

def match_templates(region_bgr, templates):
    # tenta template matching em múltiplas escalas
    found = []
    reg_gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    for key, tmpl in templates.items():
        tgray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        for scale in [0.8, 1.0, 1.2]:
            h, w = tgray.shape[:2]
            ths = cv2.resize(tgray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            if reg_gray.shape[0] < ths.shape[0] or reg_gray.shape[1] < ths.shape[1]:
                continue
            res = cv2.matchTemplate(reg_gray, ths, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.5:  # limiar bruto; refinado na classificação
                found.append((key, max_val))
    # agrega maior score por key
    best = {}
    for key, score in found:
        best[key] = max(best.get(key, 0.0), float(score))
    return best  # dict: {template_key: score}

def classify(card_img, rules, templates):
    # Extraímos duas sub-regiões heurísticas:
    h, w = card_img.shape[:2]
    # Área do ícone: quadrado alto à esquerda
    icon_roi = card_img[int(h*0.10):int(h*0.72), int(w*0.07):int(w*0.40)]
    # Área do texto/valor na base (ex.: "+300" ou "0/120K")
    text_roi = card_img[int(h*0.70):int(h*0.97), int(w*0.08):int(w*0.92)]

    text, ocr_conf = ocr_text(text_roi)
    matches = match_templates(icon_roi, templates)

    # decide bom/ruim por regras
    tscore = max(matches.values()) if matches else 0.0
    matched_icons = [k for k,v in matches.items() if v >= rules["thresholds"]["template_score"]]

    def any_in(needles, hay):
        hay_low = hay.lower()
        return any(n.lower() in hay_low for n in needles)

    is_good = False
    is_bad = False

    # por ícone
    if matched_icons:
        if any(icon in matched_icons for icon in rules.get("good", {}).get("icons", [])):
            is_good = True
        if any(icon in matched_icons for icon in rules.get("bad", {}).get("icons", [])):
            is_bad = True
    # por texto (se OCR confiável o bastante)
    if ocr_conf >= rules["thresholds"]["ocr_confidence"]:
        if any_in(rules.get("good", {}).get("text_contains", []), text):
            is_good = True
        if any_in(rules.get("bad", {}).get("text_contains", []), text):
            is_bad = True

    # empates: se marcar ambos, prioriza ruim apenas se não houver ícone bom
    label = "unknown"
    if is_good and not is_bad:
        label = "good"
    elif is_bad and not is_good:
        label = "bad"
    elif is_good and is_bad:
        label = "good" if any(icon in matched_icons for icon in rules.get("good", {}).get("icons", [])) else "bad"

    debug = {
        "text": text,
        "ocr_conf": round(ocr_conf, 3),
        "icons": matches,  # {name: score}
        "top_icon": sorted(matches.items(), key=lambda kv: kv[1], reverse=True)[:2],
        "top_icon_score": round(tscore, 3),
        "label": label
    }
    return label, debug

def notify(title, subtitle):
    try:
        Notifier.notify(subtitle, title=title, sound="default", group="mission-watcher")
    except Exception:
        pass

def log_event(prefix, data):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(LOG_DIR, f"{prefix}-{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    return path

def main_loop():
    rules = load_rules()
    templates = load_templates()
    bbox = load_config()
    poll_seconds = 5 * 60  # checar a cada 5 min

    print("Iniciando watcher (Ctrl+C para sair)…")
    with mss() as sct:
        while True:
            frame = grab_frame(sct, bbox)
            cards = find_yellow_cards(frame)

            results = []
            for (x,y,cw,ch) in cards:
                card = frame[y:y+ch, x:x+cw]
                label, dbg = classify(card, rules, templates)
                results.append({"rect": (x,y,cw,ch), "label": label, "debug": dbg})

            # heurística: o card “da esquerda” é o alvo do refresh
            left_label = results[0]["label"] if results else "unknown"
            right_label = results[1]["label"] if len(results) > 1 else "unknown"

            summary = {
                "left": left_label,
                "right": right_label,
                "details": results
            }
            log_path = log_event("scan", summary)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] left={left_label} right={right_label}  -> {log_path}")

            if right_label == "good":
                notify("Missão BOA detectada ✅", "Abra o scrcpy e aceite manualmente.")
            elif right_label == "bad":
                notify("Missão ruim detectada ❌", "Considere dar refresh manualmente.")

            time.sleep(poll_seconds)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate_bounding_box()
    else:
        main_loop()
