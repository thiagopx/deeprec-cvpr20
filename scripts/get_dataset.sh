chmod +x scripts/gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1kKxC_otX00lUs1obLHGMnJda8GvEmlJM /tmp/datasets.zip
unzip /tmp/datasets.zip -d /tmp
mv /tmp/datasets/ datasets