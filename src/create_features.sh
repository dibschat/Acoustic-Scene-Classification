#shell script to call the create_feature.py function for all the different features


echo $"mono"
python3 create_feature.py mfcc mfcc
echo $"mfcc"
python3 create_feature.py mel mono
echo $"left"
python3 create_feature.py mel left
echo $"right"
python3 create_feature.py mel right
echo $"mid"
python3 create_feature.py mel mid
echo $"side"
python3 create_feature.py mel side
echo $"harmonic"
python3 create_feature.py mel harmonic
echo $"percussive"
python3 create_feature.py mel percussive