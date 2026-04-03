#!/bin/bash
# Install XGBoost prerequisites for macOS

echo "=================================================================="
echo "Installing XGBoost Prerequisites for macOS"
echo "=================================================================="
echo ""
echo "This script will install OpenMP (libomp) which is required for"
echo "XGBoost to work on macOS."
echo ""
echo "This may take 5-10 minutes as it needs to install cmake first."
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew is not installed."
    echo ""
    echo "Install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew is installed"
echo ""

# Check if libomp is already installed
if brew list libomp &> /dev/null; then
    echo "✅ libomp is already installed"
    echo ""
    echo "Testing XGBoost import..."
    if python3 -c "import xgboost; print(f'✅ XGBoost {xgboost.__version__} loaded successfully')" 2>/dev/null; then
        echo ""
        echo "🎉 XGBoost is working! No action needed."
        exit 0
    else
        echo "⚠️  libomp installed but XGBoost still not loading"
        echo "   Try reinstalling XGBoost:"
        echo "   pip uninstall xgboost && pip install xgboost"
        exit 1
    fi
fi

echo "📦 Installing libomp (this may take 5-10 minutes)..."
echo ""

# Disable auto-update to speed up installation
export HOMEBREW_NO_AUTO_UPDATE=1

# Install libomp
if brew install libomp; then
    echo ""
    echo "✅ libomp installed successfully"
    echo ""
    echo "Testing XGBoost import..."
    if python3 -c "import xgboost; print(f'✅ XGBoost {xgboost.__version__} loaded successfully')" 2>/dev/null; then
        echo ""
        echo "🎉 SUCCESS! XGBoost is now working."
        echo ""
        echo "You can now run ML training:"
        echo "  python3 ml/test_ml_pipeline.py"
    else
        echo ""
        echo "⚠️  libomp installed but XGBoost still not loading"
        echo ""
        echo "Try these steps:"
        echo "1. Restart your terminal"
        echo "2. Reinstall XGBoost:"
        echo "   pip uninstall xgboost"
        echo "   pip install xgboost"
        echo "3. Test again:"
        echo "   python3 -c 'import xgboost; print(xgboost.__version__)'"
    fi
else
    echo ""
    echo "❌ Failed to install libomp"
    echo ""
    echo "Try manually:"
    echo "  brew install libomp"
    exit 1
fi
