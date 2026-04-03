# Contributing to StockSquad

Thank you for your interest in contributing to StockSquad! This document provides guidelines for contributing to the project.

## 🎯 Project Goals

StockSquad is designed as an educational project to demonstrate:
- Multi-agent orchestration with Azure AI Foundry
- Production-ready ML pipelines for financial analysis
- Proper software engineering practices for AI systems

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/StockSquad.git
   cd StockSquad
   ```
3. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up your environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

## 📝 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

### Testing

- Test your changes before submitting
- For ML changes, run backtests to verify performance
- Include example output in PR description

### Commit Messages

Use clear, descriptive commit messages:
```
Good: "Add ensemble predictor with voting strategy"
Bad: "fix stuff"
```

## 🔍 Areas for Contribution

### High Priority
- **Additional ML Models**: Try other algorithms (LSTM, Transformer, etc.)
- **New Agents**: Implement OptionsAgent, MacroAgent, InsiderAgent
- **Backtesting Improvements**: Add more metrics, portfolio simulation
- **Data Sources**: Integrate additional data providers

### Medium Priority
- **Documentation**: Improve setup guides, add tutorials
- **Testing**: Add unit tests, integration tests
- **Performance**: Optimize agent execution speed
- **UI/UX**: Improve CLI output, add web interface

### Low Priority
- **Code Quality**: Refactoring, type hints, linting
- **Examples**: Add more usage examples
- **Deployment**: Docker, cloud deployment guides

## 🛡️ Important Rules

### ⚠️ Never Commit:
- API keys, tokens, or credentials
- Large binary files (models > 10MB)
- Database files or training data
- Personal analysis results
- `.env` files

### ✅ Always:
- Update `.env.example` if adding new config
- Add your changes to README if user-facing
- Test on at least one ticker before submitting
- Include docstrings and type hints

## 📊 ML Contributions

If contributing ML improvements:

1. **Document your approach** in the PR
2. **Include backtest results:**
   ```
   Model: YourModel
   Ticker: AAPL (2024-01-01 to 2024-12-31)
   Win Rate: X%
   Profit Factor: X.XX
   Prediction Accuracy: X%
   ```
3. **Compare to baseline** (XGBoost or Ensemble Unanimous)
4. **Explain why** your approach is better

## 🐛 Reporting Issues

When reporting bugs, include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
- Error messages (full traceback)
- Example ticker/date range if relevant

### Issue Template

```markdown
**Bug Description:**
Clear description of the bug

**To Reproduce:**
1. Run command: `python3 main.py analyze AAPL`
2. See error

**Expected Behavior:**
Should return analysis report

**Environment:**
- Python: 3.11.5
- OS: macOS 14.0
- Branch: main

**Error Output:**
```
Paste full error message here
```
```

## 🎨 Feature Requests

Feature requests are welcome! Please include:
- **Use case**: Why is this feature useful?
- **Proposed solution**: How would it work?
- **Alternatives**: What other approaches did you consider?
- **Impact**: How many users would benefit?

## 🔄 Pull Request Process

1. **Ensure your code works:**
   - No syntax errors
   - Passes existing tests
   - Works on at least one test case

2. **Update documentation:**
   - README.md (if user-facing)
   - Docstrings in code
   - Comments for complex logic

3. **Create detailed PR description:**
   ```markdown
   ## What does this PR do?
   Brief description

   ## Why is this needed?
   Problem it solves

   ## How was it tested?
   Test commands and results

   ## Screenshots/Output
   Example output if applicable
   ```

4. **Request review**

5. **Address feedback**

## 💰 Financial Analysis Disclaimer

Contributions involving financial analysis or trading strategies must include:
- ⚠️ Disclaimer that results are for educational purposes only
- 📊 Backtest results showing historical performance
- 🚨 Warning about risks of real trading
- 📝 Clear documentation of assumptions

**Never promote:**
- "Get rich quick" strategies
- Guaranteed returns
- Financial advice
- Real money trading without proper risk warnings

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Welcome newcomers
- No spam or self-promotion
- Keep discussions professional

## 📚 Resources for Contributors

- [Azure OpenAI Assistants API](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant)
- [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-services/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)

## 🙋 Questions?

- Open an issue with the "question" label
- Check existing issues first
- Provide context and what you've tried

## 🎉 Recognition

Contributors will be:
- Added to a CONTRIBUTORS file
- Mentioned in release notes
- Appreciated greatly!

---

Thank you for helping make StockSquad better! 🚀
