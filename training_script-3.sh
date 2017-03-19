python -u fit_model.py linearmodel configs/imdb.json -embedding data/depw2v.pkl  &> linearmodel-imdb.log &
python -u fit_model.py ttbb configs/imdb.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl   &> ttbb-imdb.log &
python -u fit_model.py linearmodel configs/agnews.json -embedding data/depw2v.pkl &> linearmodel-agnews.log &
python -u fit_model.py ttbb configs/agnews.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl  &> ttbb-agnews.log &
python -u fit_model.py linearmodel configs/amazon.json -embedding data/depw2v.pkl &> linearmodel-amazon.log &
python -u fit_model.py ttbb configs/amazon.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl &> ttbb-amazon.log &
