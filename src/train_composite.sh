#shell script to call the train.py function for all the different features


echo $"mono"
python3 train.py full mono
echo $"mfcc"
python3 train.py full mfcc
echo $"left"
python3 train.py full left
echo $"right"
python3 train.py full right
echo $"mid"
python3 train.py full mid
echo $"side"
python3 train.py full side
echo $"harmonic"
python3 train.py full harmonic
echo $"percussive"
python3 train.py full percussive