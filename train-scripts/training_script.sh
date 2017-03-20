python -u fit_model.py fasttextpretrain configs/imdb.json -embedding data/depw2v.pkl  &> fasttestpretrain-imdb.log &
python -u fit_model.py ttbbtune configs/imdb.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl   &> ttbbtune-imdb.log &
python -u fit_model.py fasttextpretrain configs/agnews.json -embedding data/depw2v.pkl &> fasttestpretrain-agnews.log &
python -u fit_model.py ttbbtune configs/agnews.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl  &> ttbbtune-agnews.log &
python -u fit_model.py fasttextpretrain configs/amazon.json -embedding data/depw2v.pkl &> fasttestpretrain-amazon.log &
python -u fit_model.py ttbbtune configs/amazon.json -embedding data/depw2v.pkl -word_freq data/word_freq.pkl &> ttbbtune-amazon.log &
