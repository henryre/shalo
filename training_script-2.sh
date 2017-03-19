python -u fit_model.py sparselm configs/imdb.json -embedding data/depw2v.pkl  &> sparselm-imdb.log &
python -u fit_model.py lstmpretrain configs/imdb.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl   &> lstmpretrain-imdb.log &
python -u fit_model.py sparselm configs/agnews.json -embedding data/depw2v.pkl &> sparselm-agnews.log &
python -u fit_model.py lstmpretrain configs/agnews.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl  &> lstmpretrain-agnews.log &
python -u fit_model.py sparselm configs/amazon.json -embedding data/depw2v.pkl &> sparselm-amazon.log &
python -u fit_model.py lstmpretrain configs/amazon.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl &> lstmpretrain-amazon.log &
