python -u fit_model.py lstmpretrain configs/imdb_lstm.json -embedding data/depw2v.pkl  &> lstmpretrain-imdb.log &
python -u fit_model.py ttbbtuneexact configs/imdb.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl   &> ttbbtuneexact-imdb.log &
python -u fit_model.py lstmpretrain configs/agnews_lstm.json -embedding data/depw2v.pkl &> lstmpretrain-agnews.log &
python -u fit_model.py ttbbtuneexact configs/agnews.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl  &> ttbbtuneexact-agnews.log &
python -u fit_model.py lstmpretrain configs/amazon_lstm.json -embedding data/depw2v.pkl &> lstmpretrain-amazon.log &
python -u fit_model.py ttbbtuneexact configs/amazon.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl &> ttbbtuneexact-amazon.log &
