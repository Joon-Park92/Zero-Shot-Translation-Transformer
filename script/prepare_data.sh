#!/bin/bash

# follwing https://google.github.io/styleguide/shell.xml#Indentation

# DOWNLOAD PARALLEL LANGUAGE FILE
# wget http://opus.nlpl.eu/download.php\?f\=OpenSubtitles/v2018/moses/en-ko.txt.zip
# mkdir en-ko
# unzip download.php?f=OpenSubtitles%2Fv2018%2Fmoses%2Fen-ko.txt.zip -d ./en-ko/

# PREPROCESS DATA
L1="OpenSubtitles.en-ko.en"
L2="OpenSubtitles.en-ko.ko"

TEST_SIZE=10000
VAL_SIZE=10000
NUM_BPE_OPERATIONS=10000
VOCABULARY_THRESHOLD=50


preprocessing () {	

  RAW_FILE=$1

  echo CLEAN $RAW_FILE

  # it works only for ascii encoding
#  sed -e "s/[^[:print:]]//g" -i $RAW_FILE

  # for utf-8 
  sed -e 's/\xc2\x91\|\xc2\x92\|\xc2\xa0\|\xe2\x80\x8e//' -i $RAW_FILE
  sed -e "s///g" -i $RAW_FILE
  sed -e "s/\s/ /g" -i $RAW_FILE
  sed -e "s/[ ]{2,}/ /g" -i $RAW_FILE

  echo SPLIT $RAW_FILE
  TEST_FILE=$RAW_FILE.test
  VAL_FILE=$RAW_FILE.val
  TRAIN_FILE=$RAW_FILE.train

  sed -n "1,$TEST_SIZE p" $RAW_FILE > $TEST_FILE
  sed -n "$(($TEST_SIZE+1)),$(($TEST_SIZE+$VAL_SIZE)) p" $RAW_FILE > $VAL_FILE
  sed -n "$(($TEST_SIZE+$VAL_SIZE+1)),$ p" $RAW_FILE > $TRAIN_FILE

  CODES_FILE=$TRAIN_FILE.codes
  VOCAB_FILE=$TRAIN_FILE.vocab
  
  echo LEARN AND APPLY BPE PROCESS
  subword-nmt learn-bpe -s $NUM_BPE_OPERATIONS < $TRAIN_FILE > $CODES_FILE
  subword-nmt apply-bpe -c $CODES_FILE < $TRAIN_FILE | subword-nmt get-vocab > $VOCAB_FILE

  for FILE in $TRAIN_FILE $VAL_FILE $TEST_FILE
    do
      subword-nmt apply-bpe -c $CODES_FILE \
        --vocabulary $VOCAB_FILE \
        --vocabulary-threshold $VOCABULARY_THRESHOLD < $FILE > $FILE.ready
    done
}


#echo SHUFFLING...
#paste -d ':' $L1 $L2 | shuf | awk -v FS=':' -v f1="$L1.shuf" -v f2="$L2.shuf" '{ print $1 > f1 ; print $2 > f2 }'

# for LANG in $L1 $L2
for LANG in $L2
  do
    preprocessing $LANG
  done

./build_dictionary.py train_file.BPE.L1 train_file.BPE.L2
