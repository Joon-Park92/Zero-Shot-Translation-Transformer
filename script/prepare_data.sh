#!/bin/bash

# PREPROCESS DATA
DOWN_DIR=/home/slayer/cloned/Zero-Shot-Translation-Transformer/script/open_subtitle2018
L1=$DOWN_DIR/OpenSubtitles.en-ko.en
L2=$DOWN_DIR/OpenSubtitles.en-ko.ko
TEST_SIZE=10000
VAL_SIZE=10000
NUM_BPE_OPERATIONS=10000
VOCABULARY_THRESHOLD=50


# DOWNLOAD PARALLEL LANGUAGE FILE
download () {
  mkdir $DOWN_DIR 
  wget http://opus.nlpl.eu/download.php\?f\=OpenSubtitles/v2018/moses/en-ko.txt.zip
  unzip download.php?f=OpenSubtitles%2Fv2018%2Fmoses%2Fen-ko.txt.zip -d $DOWN_DIR
}

shuffle () {
  echo SHUFFLING...
  paste -d ':' $L1 $L2 | shuf | awk -v FS=':' -v f1="$L1.shuf" -v f2="$L2.shuf" '{ print $1 > f1 ; print $2 > f2 }'
}

preprocessing () {	

  echo CLEAN $1

  # REMOVE INVALID UTF-8 CHARACTERS
  sed -e 's/[^[:print:]]//g' -i $1
  sed -e "s///g" -i $1
  sed -e "s/\s/ /g" -i $1
  sed -e "s/[ ]{2,}/ /g" -i $1

  echo SPLIT $1
  TEST_FILE=$1.test
  VAL_FILE=$1.val
  TRAIN_FILE=$1.train

  sed -n "1,$TEST_SIZE p" $1 > $TEST_FILE
  sed -n "$(($TEST_SIZE+1)),$(($TEST_SIZE+$VAL_SIZE)) p" $1 > $VAL_FILE
  sed -n "$(($TEST_SIZE+$VAL_SIZE+1)),$ p" $1 > $TRAIN_FILE

  CODES_FILE=$TRAIN_FILE.codes
  VOCAB_FILE=$TRAIN_FILE.vocab
  
  echo LEARN AND APPLY BPE PROCESS
  subword-nmt learn-bpe -s $NUM_BPE_OPERATIONS < $TRAIN_FILE > $CODES_FILE
  subword-nmt apply-bpe -c $CODES_FILE < $TRAIN_FILE | subword-nmt get-vocab > $VOCAB_FILE

  subword-nmt apply-bpe -c $CODES_FILE \
    --vocabulary $VOCAB_FILE \
    --vocabulary-threshold $VOCABULARY_THRESHOLD < $TRAIN_FILE > $TRAIN_FILE.ready

  subword-nmt apply-bpe -c $CODES_FILE \
    --vocabulary $VOCAB_FILE \
    --vocabulary-threshold $VOCABULARY_THRESHOLD < $VAL_FILE > $VAL_FILE.ready

  subword-nmt apply-bpe -c $CODES_FILE \
    --vocabulary $VOCAB_FILE \
    --vocabulary-threshold $VOCABULARY_THRESHOLD < $TEST_FILE > $TEST_FILE.ready
}

main () {

  download 
  shuffle
  preprocessing $L1.shuf
  preprocessing $L2.shuf
  # ./build_dictionary.py 
}

main
