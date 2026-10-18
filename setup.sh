#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/

# Create Streamlit config
echo "\
[general]\n\
headless = true\n\
\n\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
" > ~/.streamlit/config.toml