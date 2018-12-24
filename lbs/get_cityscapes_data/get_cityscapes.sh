#!/usr/bin/env bash

src=/data/nikita/lbs
wget --directory-prefix=$src/lbs/get_cityscapes_data --keep-session-cookies --save-cookies=$src/lbs/get_cityscapes_data/cookies.txt --post-data 'username=nikitavemuri@berkeley.edu&password=cityscapes&submit=Login' https://www.cityscapes-dataset.com/login/
wget --directory-prefix=$src/lbs/get_cityscapes_data --load-cookies $src/lbs/get_cityscapes_data/cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --directory-prefix=$src/lbs/get_cityscapes_data --load-cookies $src/lbs/get_cityscapes_data/cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip $src/lbs/get_cityscapes_data/gtFine_trainvaltest.zip -d $src/lbs/get_cityscapes_data
rm $src/lbs/get_cityscapes_data/README $src/lbs/get_cityscapes_data/license.txt
unzip $src/lbs/get_cityscapes_data/leftImg8bit_trainvaltest.zip -d $src/lbs/get_cityscapes_data
rm $src/lbs/get_cityscapes_data/README $src/lbs/get_cityscapes_data/license.txt
python3 $src/lbs/get_cityscapes_data/prepare_data.py $src/lbs/get_cityscapes_data/gtFine/
bash $src/lbs/get_cityscapes_data/create_lists.sh
rm $src/lbs/get_cityscapes_data/cookies.txt $src/lbs/get_cityscapes_data/index.html $src/lbs/get_cityscapes_data/gtFine_trainvaltest.zip $src/lbs/get_cityscapes_data/leftImg8bit_trainvaltest.zip
mkdir $src/data/cityscapes
mv $src/lbs/get_cityscapes_data/gtFine $src/lbs/get_cityscapes_data/leftImg8bit $src/lbs/get_cityscapes_data/train_images.txt $src/lbs/get_cityscapes_data/train_labels.txt $src/lbs/get_cityscapes_data/val_images.txt $src/lbs/get_cityscapes_data/val_labels.txt $src/lbs/get_cityscapes_data/test_images.txt $src/data/cityscapes
cp $src/lbs/get_cityscapes_data/info.json $src/data/cityscapes
