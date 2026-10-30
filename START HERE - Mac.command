#!/bin/bash
# Double-click to install (first time only) and open the app. Every step is skipped when already done:
# AI key (asked once) -> uv (installs Python + libraries, no admin rights) -> Python 3.11 in .venv -> libraries -> app.
# --change-key (used by "Change AI Key - Mac.command") only asks for a new key and saves it.
cd "$(dirname "$0")" || exit 1

step() { printf '\n\033[32m== %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$1"; read -rp "Press Enter to close."; exit 1; }

save_key() {  # keep any other settings already in .env; only the key line changes
  src=.env.example; [ -f .env ] && src=.env
  sed "s|^LLM_API_KEY=.*|LLM_API_KEY=$1|" "$src" > .env.tmp && mv .env.tmp .env
}

# 1. AI key: asked first (so the install can run unattended), checked with DeepSeek before saving
if [ "$1" = "--change-key" ] || [ ! -f .env ]; then
  step "AI key"
  echo "Paste your DeepSeek API key and press Enter (get one at https://platform.deepseek.com/api_keys)."
  echo "Or just press Enter to skip: the app works without AI. You can add a key later with 'Change AI Key - Mac.command'."
  while true; do
    read -rp "API key: " KEY
    KEY=$(printf '%s' "$KEY" | tr -d '[:space:]')
    if [ -z "$KEY" ]; then
      [ -f .env ] || save_key ""
      echo "Skipped. AI features stay off."
      break
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -H "Authorization: Bearer $KEY" https://api.deepseek.com/models)
    if [ "$CODE" = "200" ]; then
      save_key "$KEY"; printf '\033[32mKey works and is saved.\033[0m\n'; break
    elif [ "$CODE" = "401" ] || [ "$CODE" = "403" ]; then
      printf '\033[33mDeepSeek rejected this key. Copy it again and paste it, or press Enter to skip.\033[0m\n'
    else
      save_key "$KEY"; printf '\033[33mCould not reach DeepSeek to check the key (no internet?). Saved it anyway.\033[0m\n'; break
    fi
  done
fi
if [ "$1" = "--change-key" ]; then
  echo; echo "Done. If the app is open, close its window and start it again with 'START HERE - Mac.command'."
  read -rp "Press Enter to close."; exit 0
fi

# 2. uv
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
if [ ! -x "$UV" ]; then
  step "Installing uv (it downloads Python and the libraries the app needs)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -x "$UV" ] || fail "Could not install uv. Check the internet connection and open 'START HERE - Mac.command' again."
fi

# 3. Python 3.11 in .venv
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  step "Setting up Python 3.11 (first run only)"
  "$UV" venv .venv --python 3.11 --seed || fail "Could not set up Python. Check the internet connection and try again."
fi

# 4. Libraries: reinstalled only when requirements.txt changes
HASH=$(shasum -a 256 requirements.txt | cut -d' ' -f1)
if [ "$(cat .venv/.installed 2>/dev/null)" != "$HASH" ]; then
  step "Installing libraries (first run takes a few minutes)"
  "$UV" pip install --python "$PY" -r requirements.txt || fail "Could not install the libraries. Check the internet connection and try again."
  echo "$HASH" > .venv/.installed
fi

# Streamlit asks for an e-mail on its very first start; answer it so the app starts straight away
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
  mkdir -p "$HOME/.streamlit" && printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

# 5. Run
step "Starting the app. It opens in your browser. Keep this window open while you use it; close it to stop."
exec "$PY" -m streamlit run app.py
