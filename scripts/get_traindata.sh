chmod +x scripts/gdown.pl
scripts/gdown.pl https://drive.google.com/file/d/1gYANaxaEfXfz473ufpXxoGoLrCODF_vh /tmp/traindata.zip
unzip /tmp/traindata.zip -d /tmp
mv /tmp/traindata traindata
