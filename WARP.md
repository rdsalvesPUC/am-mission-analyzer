# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The am-mission-analyzer is an automatic analyzer for personal missions in Alliance Mobilization. It monitors and notifies users when desired missions become available.

## Current State

This is an early-stage project repository that currently contains:
- Initial README.md with project description
- MIT License
- Git repository setup

## Development Setup

Since this project is in initial stages, the development setup will depend on the technology stack chosen. Common setup patterns would include:

### For Python Development:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Run tests (once test framework is set up)
python -m pytest

# Run the application
python main.py
```

### For Node.js Development:
```bash
# Install dependencies
npm install

# Run tests
npm test

# Start development server
npm run dev

# Build for production
npm run build
```

### For Go Development:
```bash
# Initialize module (if not done)
go mod init am-mission-analyzer

# Install dependencies
go mod tidy

# Run tests
go test ./...

# Build the application
go build

# Run the application
./am-mission-analyzer
```

## Anticipated Architecture

Based on the project description, this application will likely need:

### Core Components:
- **Mission Monitor**: Service to check Alliance Mobilization for available missions
- **User Preferences**: System to store which missions users are interested in
- **Notification System**: Alert users when desired missions become available
- **Data Storage**: Persistence layer for user preferences and mission data
- **API Client**: Interface with Alliance Mobilization services

### Potential Structure:
```
/
├── src/                    # Source code
│   ├── monitor/           # Mission monitoring logic
│   ├── notifications/     # Notification system
│   ├── storage/          # Data persistence
│   └── api/              # API client and server
├── tests/                 # Test files
├── config/               # Configuration files
└── docs/                 # Documentation
```

## Git Workflow

```bash
# Check current status
git status

# View recent changes
git --no-pager log --oneline -10

# Create feature branch
git checkout -b feature/mission-monitor

# Commit changes
git add .
git commit -m "Add mission monitoring functionality"

# Push changes
git push origin feature/mission-monitor
```

## Future Development Notes

When implementing this project, consider:

1. **Rate Limiting**: Implement appropriate delays when polling Alliance Mobilization to avoid overwhelming their servers
2. **Error Handling**: Robust error handling for network requests and API responses
3. **Configuration**: Allow users to configure check intervals and notification preferences
4. **Privacy**: Handle user credentials and preferences securely
5. **Logging**: Implement comprehensive logging for debugging and monitoring

## Contributing

This project uses the MIT License. All contributions should maintain compatibility with this license.
