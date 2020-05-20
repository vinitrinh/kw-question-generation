#!/usr/bin/env bash 

pip install -r requirements.txt

# get stanza english model
python - <<-EOF
import stanza
stanza.download('en')
EOF

# install sent2vec library
git clone https://github.com/epfml/sent2vec.git
cd sent2vec
pip install .
cd ..

gdown https://drive.google.com/uc?id=0B6VhzidiLvjSa19uYWlLUEkzX3c

