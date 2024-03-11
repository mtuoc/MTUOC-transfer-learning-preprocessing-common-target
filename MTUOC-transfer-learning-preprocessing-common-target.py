#    MTUOC-transfer-learning-preprocessing-common-target
#    Copyright (C) 2024 Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
from datetime import datetime
import os
import glob
import codecs
import importlib
import re
import random

from shutil import copyfile

import yaml
from yaml import load, dump

from itertools import (takewhile,repeat)

import sentencepiece as spm

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
    
def findEMAILs(string): 
    email=re.findall('\S+@\S+', string)   
    return email
    
def findURLs(text):
    # Regular expression for identifying URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Find all matches using the regular expression
    matches = re.findall(url_pattern, text)
    return(matches)
    
def replace_EMAILs(string,code="@EMAIL@"):
    EMAILs=findEMAILs(string)
    cont=0
    for EMAIL in EMAILs:
        string=string.replace(EMAIL,code)
    return(string)

def replace_URLs(string,code="@URL@"):
    URLs=findURLs(string)
    cont=0
    for URL in URLs:
        string=string.replace(URL,code)
    return(string)

def split_corpusA(filename,valsize,evalsize,slcode,tlcode):
    count=rawincount(filename)
    numlinestrain=count-valsize-evalsize
    numlinestrain2=numlinestrain
    if numlinestrain<0: numlinestrain2=0
    entrada=codecs.open(filename,"r",encoding="utf-8")
    filenametrain="trainA-"+slcode+"-"+tlcode+".txt"
    sortidaTrain=codecs.open(filenametrain,"w",encoding="utf-8")
    filenameval="valA-"+slcode+"-"+tlcode+".txt"
    sortidaVal=codecs.open(filenameval,"w",encoding="utf-8")
    filenameeval="evalA-"+slcode+"-"+tlcode+".txt"
    sortidaEval=codecs.open(filenameeval,"w",encoding="utf-8")
    cont=0
    for linia in entrada:
        if cont < numlinestrain:
            sortidaTrain.write(linia)
        elif cont>= numlinestrain2 and cont < numlinestrain2+valsize:
            sortidaVal.write(linia)
        else:
            sortidaEval.write(linia)
        cont+=1
    sortidaTrain.close()
    sortidaVal.close()
    sortidaEval.close()

def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

def split_corpusB(filename,valsize,evalsize,slcode,tlcode):
    count=rawincount(filename)
    numlinestrain=count-valsize-evalsize
    numlinestrain2=numlinestrain
    if numlinestrain<0: numlinestrain2=0
    entrada=codecs.open(filename,"r",encoding="utf-8")
    filenametrain="trainB-"+slcode+"-"+tlcode+".txt"
    sortidaTrain=codecs.open(filenametrain,"w",encoding="utf-8")
    filenameval="valB-"+slcode+"-"+tlcode+".txt"
    sortidaVal=codecs.open(filenameval,"w",encoding="utf-8")
    filenameeval="evalB-"+slcode+"-"+tlcode+".txt"
    sortidaEval=codecs.open(filenameeval,"w",encoding="utf-8")
    cont=0
    for linia in entrada:
        if cont < numlinestrain:
            sortidaTrain.write(linia)
        elif cont>= numlinestrain2 and cont < numlinestrain2+valsize:
            sortidaVal.write(linia)
        else:
            sortidaEval.write(linia)
        cont+=1
    sortidaTrain.close()
    sortidaVal.close()
    sortidaEval.close()
    
def sentencepiece_train(corpusL1,corpusL3,corpusL2,L1code2,L3code2,L2code2,SP_MODEL_PREFIX="spmodel",MODEL_TYPE="bpe",VOCAB_SIZE=32000,CHARACTER_COVERAGE=1,INPUT_SENTENCE_SIZE=1000000,SPLIT_DIGITS=True,CONTROL_SYMBOLS="",USER_DEFINED_SYMBOLS=""):
    options=[]
    if SPLIT_DIGITS:
        options.append("--split_digits=true")
    else:
        options.append("--split_digits=false")
    if not CONTROL_SYMBOLS=="":
        options.append("--control_symbols=\""+CONTROL_SYMBOLS+"\"")
    if not USER_DEFINED_SYMBOLS=="":
        options.append("--user_defined_symbols=\""+USER_DEFINED_SYMBOLS+"\"") 
    options=" ".join(options)
    if True:
        command = "cat "+corpusL1+" "+corpusL3+" "+corpusL2+" | shuf > train"
        os.system(command)        
        command="spm_train --input=train --model_prefix="+SP_MODEL_PREFIX+" --model_type="+MODEL_TYPE+" --vocab_size="+str(VOCAB_SIZE)+" --character_coverage="+str(CHARACTER_COVERAGE)+" --split_digits --input_sentence_size="+str(INPUT_SENTENCE_SIZE)+" "+options
        print(command)
        os.system(command)
        command="spm_encode --model="+SP_MODEL_PREFIX+".model --generate_vocabulary < "+corpusL1+" > vocab_file."+L1code2
        os.system(command)
        command="spm_encode --model="+SP_MODEL_PREFIX+".model --generate_vocabulary < "+corpusL3+" > vocab_file."+L3code2
        os.system(command)
        command="spm_encode --model="+SP_MODEL_PREFIX+".model --generate_vocabulary < "+corpusL2+" > vocab_file."+L2code2
        os.system(command)
        
def sentencepiece_encode(corpusPre,OUTFILE,SP_MODEL,VOCABULARY,VOCABULARY_THRESHOLD=50, EOS=True, BOS=True):
    if EOS and BOS:
        extraoptions="--extra_options eos:bos"
    elif EOS:
        extraoptions="--extra_options eos"
    elif BOS:
        extraoptions="--extra_options bos"
    else:
        extraoptions=""
    command="spm_encode --model="+SP_MODEL+" "+extraoptions+" --vocabulary="+VOCABULARY+" --vocabulary_threshold="+str(VOCABULARY_THRESHOLD)+" < "+corpusPre+" > "+OUTFILE
    os.system(command)

stream = open('config-transfer-learning-preprocessing-common-target.yaml', 'r',encoding="utf-8")
config=yaml.load(stream, Loader=yaml.FullLoader)
MTUOC=config["MTUOC"]
sys.path.append(MTUOC)
from MTUOC_guided_alignment_eflomal import guided_alignment_eflomal
from MTUOC_guided_alignment_fast_align import guided_alignment_fast_align
from MTUOC_check_guided_alignment import check_guided_alignment

from MTUOC_train_truecaser import TC_Trainer
from MTUOC_truecaser import Truecaser
from MTUOC_splitnumbers import splitnumbers

VERBOSE=config["VERBOSE"]
LOGFILE=config["LOG_FILE"]

corpusA=config["corpusA"]
#L3-L2 corpus
corpusB=config["corpusB"]
#L1-L2 corpus
valsizeA=config["valsizeA"]
evalsizeA=config["evalsizeA"]
valsizeB=config["valsizeB"]
evalsizeB=config["evalsizeB"]
L1code3=config["L1code3"]
L1code2=config["L1code2"]
L3code3=config["L3code3"]
L3code2=config["L3code2"]
L2code3=config["L2code3"]
L2code2=config["L2code2"]


L1_DICT=config["L1_DICT"]
L3_DICT=config["L3_DICT"]
L2_DICT=config["L2_DICT"]
#state None or null.dict if not word form dictionary available for that languages

L1_TOKENIZER=config["L1_TOKENIZER"]
TOKENIZE_L1=config["TOKENIZE_L1"]
L2_TOKENIZER=config["L2_TOKENIZER"]
TOKENIZE_L2=config["TOKENIZE_L2"]
L3_TOKENIZER=config["L3_TOKENIZER"]
TOKENIZE_L3=config["TOKENIZE_L3"]

if L1_TOKENIZER==None: TOKENIZE_L1=False
if L3_TOKENIZER==None: TOKENIZE_L3=False
if L2_TOKENIZER==None: TOKENIZE_L2=False

###PREPARATION
REPLACE_EMAILS=config["REPLACE_EMAILS"]
EMAIL_CODE=config["EMAIL_CODE"]
REPLACE_URLS=config["REPLACE_URLS"]
URL_CODE=config["URL_CODE"]

TRAIN_L1_TRUECASER=config["TRAIN_L1_TRUECASER"]
TRUECASE_L1=config["TRUECASE_L1"]
L1_TC_MODEL=config["L1_TC_MODEL"]
if L1_TC_MODEL=="auto":
    L1_TC_MODEL="tc."+L1code2

TRAIN_L3_TRUECASER=config["TRAIN_L3_TRUECASER"]
TRUECASE_L3=config["TRUECASE_L3"]
L3_TC_MODEL=config["L3_TC_MODEL"]
if L3_TC_MODEL=="auto":
    L3_TC_MODEL="tc."+L3code2

TRAIN_L2_TRUECASER=config["TRAIN_L2_TRUECASER"]
TRUECASE_L2=config["TRUECASE_L2"]
L2_TC_MODEL=config["L2_TC_MODEL"]
if L2_TC_MODEL=="auto":
    L2_TC_MODEL="tc."+L2code2

CLEAN=config["CLEAN"]
MIN_TOK=config["MIN_TOK"]
MAX_TOK=config["MAX_TOK"]

MIN_CHAR=config["MIN_CHAR"]
MAX_CHAR=config["MAX_CHAR"]


TRAIN_SENTENCEPIECE=config["TRAIN_SENTENCEPIECE"]
SAMPLE_SIZE=config["SAMPLE_SIZE"]
bos=config["bos"]
#<s> or None
eos=config["eos"]
#</s> or None
bosSP=True
eosSP=True
if bos=="None": bosSP=False
if eos=="None": eosSP=False
JOIN_LANGUAGES=config["JOIN_LANGUAGES"]
SPLIT_DIGITS=config["SPLIT_DIGITS"]
VOCABULARY_THRESHOLD=config["VOCABULARY_THRESHOLD"]

CONTROL_SYMBOLS=config["CONTROL_SYMBOLS"]
USER_DEFINED_SYMBOLS=config["USER_DEFINED_SYMBOLS"]
SP_MODEL_PREFIX=config["SP_MODEL_PREFIX"]
MODEL_TYPE=config["MODEL_TYPE"]
#one of unigram, bpe, char, word
VOCAB_SIZE=config["VOCAB_SIZE"]
CHARACTER_COVERAGE=config["CHARACTER_COVERAGE"]
CHARACTER_COVERAGE_SL=config["CHARACTER_COVERAGE_SL"]
CHARACTER_COVERAGE_TL=config["CHARACTER_COVERAGE_TL"]
INPUT_SENTENCE_SIZE=config["INPUT_SENTENCE_SIZE"]

#GUIDED ALIGNMENT
#TRAIN CORPUS
GUIDED_ALIGNMENT=config["GUIDED_ALIGNMENT"]
ALIGNER=config["ALIGNER"]
#one of eflomal, fast_align, simalign, awesome
DELETE_EXISTING=config["DELETE_EXISTING"]
SPLIT_LIMIT=config["SPLIT_LIMIT"]
#For efomal, max number of segments to align at a time

#VALID CORPUS
GUIDED_ALIGNMENT_VALID=config["GUIDED_ALIGNMENT_VALID"]
ALIGNER_VALID=config["ALIGNER_VALID"]
#one of eflomal, fast_align, simalign, awesome
DELETE_EXISTING_VALID=config["DELETE_EXISTING_VALID"]

DELETE_TEMP=config["DELETE_TEMP"]

if VERBOSE:
    logfile=codecs.open(LOGFILE,"w",encoding="utf-8")

#SPLITTING CORPUS A
if VERBOSE:
    cadena="Splitting corpus A: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
split_corpusA(corpusA,valsizeA,evalsizeA,L3code3,L2code3)

trainCorpus="trainA-"+L3code3+"-"+L2code3+".txt"
valCorpus="valA-"+L3code3+"-"+L2code3+".txt"
evalCorpus="evalA-"+L3code3+"-"+L2code3+".txt"
trainPreCorpus="trainA-pre-"+L3code3+"-"+L2code3+".txt"
valPreCorpus="valA-pre-"+L3code3+"-"+L2code3+".txt"
evalSL="evalA."+L3code2
evalTL="evalA."+L2code2
entrada=codecs.open(evalCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open(evalSL,"w",encoding="utf-8")
sortidaTL=codecs.open(evalTL,"w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()

#SPLITTING CORPUS B
if VERBOSE:
    cadena="Splitting corpus B: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
split_corpusB(corpusB,valsizeB,evalsizeB,L1code3,L2code3)

trainCorpus="trainB-"+L1code3+"-"+L2code3+".txt"
valCorpus="valB"+L1code3+"-"+L2code3+".txt"
evalCorpus="evalB-"+L1code3+"-"+L2code3+".txt"
trainPreCorpus="trainB-pre-"+L1code3+"-"+L2code3+".txt"
valPreCorpus="valB-pre-"+L1code3+"-"+L2code3+".txt"
evalSL="evalB."+L1code2
evalTL="evalB."+L2code2
entrada=codecs.open(evalCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open(evalSL,"w",encoding="utf-8")
sortidaTL=codecs.open(evalTL,"w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()

trainCorpus="trainA-"+L3code3+"-"+L2code3+".txt"
entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open("trainASL.temp","w",encoding="utf-8")
sortidaTL=codecs.open("trainATL.temp","w",encoding="utf-8")
#sortidaW=codecs.open(train_weightsFile,"w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
        #if len(camps)>=3:
        #    sortidaW.write(camps[2]+"\n")
        #else:
        #    sortidaW.write("\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()

trainCorpus="trainB-"+L1code3+"-"+L2code3+".txt"
entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open("trainBSL.temp","w",encoding="utf-8")
sortidaTL=codecs.open("trainBTL.temp","w",encoding="utf-8")
#sortidaW=codecs.open(train_weightsFile,"w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
        #if len(camps)>=3:
        #    sortidaW.write(camps[2]+"\n")
        #else:
        #    sortidaW.write("\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()

#TRUECASERS




if TRAIN_L1_TRUECASER: #CORPUS B  SL
    if VERBOSE:
        cadena="Training L1 Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    SLTrainer=TC_Trainer(MTUOC, L1_TC_MODEL, "trainBSL.temp", L1_DICT, L1_TOKENIZER)
    SLTrainer.train_truecaser()

if TRAIN_L3_TRUECASER:# CORPUS A  SL
    if VERBOSE:
        cadena="Training L3 Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    SLTrainer=TC_Trainer(MTUOC, L3_TC_MODEL, "trainASL.temp", L3_DICT, L3_TOKENIZER)
    SLTrainer.train_truecaser()  

#concateniating trainATL.temp and trainBTL.temp

outconcat=codecs.open("tempconcat.temp","w",encoding="utf-8")

entradaconcat=codecs.open("trainATL.temp","r",encoding="utf-8")
for linia in entradaconcat:
    linia=linia.rstrip()
    outconcat.write(linia+"\n")
entradaconcat.close()

entradaconcat=codecs.open("trainBTL.temp","r",encoding="utf-8")
for linia in entradaconcat:
    linia=linia.rstrip()
    outconcat.write(linia+"\n")
entradaconcat.close()

if TRAIN_L2_TRUECASER: #CORPUS A TL i CORPUS B TL
    if VERBOSE:
        cadena="Training L2 Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    SLTrainer=TC_Trainer(MTUOC, L2_TC_MODEL, "tempconcat.temp", L2_DICT, L2_TOKENIZER)
    SLTrainer.train_truecaser()  

if TRUECASE_L1:
    truecaserL1=Truecaser()
    truecaserL1.set_MTUOCPath(MTUOC)
    truecaserL1.set_tokenizer(L1_TOKENIZER)
    truecaserL1.set_tc_model(L1_TC_MODEL)

if TRUECASE_L3:
    truecaserL3=Truecaser()
    truecaserL3.set_MTUOCPath(MTUOC)
    truecaserL3.set_tokenizer(L3_TOKENIZER)
    truecaserL3.set_tc_model(L3_TC_MODEL)
    
if TRUECASE_L2:
    truecaserL2=Truecaser()
    truecaserL2.set_MTUOCPath(MTUOC)
    truecaserL2.set_tokenizer(L2_TOKENIZER)
    truecaserL2.set_tc_model(L2_TC_MODEL)

if not L1_TOKENIZER==None:
    L1_TOKENIZER=MTUOC+"/"+L1_TOKENIZER
    if not L1_TOKENIZER.endswith(".py"): L1_TOKENIZER=L1_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', L1_TOKENIZER)
    tokenizerL1mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerL1mod)
    tokenizerL1=tokenizerL1mod.Tokenizer()
    
if not L3_TOKENIZER==None:
    L3_TOKENIZER=MTUOC+"/"+L3_TOKENIZER
    if not L3_TOKENIZER.endswith(".py"): L3_TOKENIZER=L3_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', L3_TOKENIZER)
    tokenizerL3mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerL3mod)
    tokenizerL3=tokenizerL3mod.Tokenizer()
    
if not L2_TOKENIZER==None:
    L2_TOKENIZER=MTUOC+"/"+L2_TOKENIZER
    if not L2_TOKENIZER.endswith(".py"): L2_TOKENIZER=L2_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', L2_TOKENIZER)
    tokenizerL2mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerL2mod)
    tokenizerL2=tokenizerL2mod.Tokenizer()

if VERBOSE:
    cadena="Preprocessing trainA corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

trainCorpus="trainA-"+L3code3+"-"+L2code3+".txt"
trainPreCorpus="trainA-pre-"+L3code3+"-"+L2code3+".txt"

entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortida=codecs.open(trainPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        #if len(camps)>=3:
        #    weight=camps[2]
        #else:
        #    weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_L3:
            toksl=tokenizerL3.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_L2:
            toktl=tokenizerL2.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_L3:
                toksl=truecaserL1.truecase(toksl)
            if TRUECASE_L2:
                toktl=truecaserL3.truecase(toktl)
            #if not weight==None:
            #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            #else:
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()

if VERBOSE:
    cadena="Preprocessing trainB corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

trainCorpus="trainB-"+L1code3+"-"+L2code3+".txt"
trainPreCorpus="trainB-pre-"+L1code3+"-"+L2code3+".txt"

entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortida=codecs.open(trainPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        #if len(camps)>=3:
        #    weight=camps[2]
        #else:
        #    weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_L1:
            toksl=tokenizerL1.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_L2:
            toktl=tokenizerL2.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_L1:
                toksl=truecaserL1.truecase(toksl)
            if TRUECASE_L2:
                toktl=truecaserL2.truecase(toktl)
            #if not weight==None:
            #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            #else:
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()

if VERBOSE:
    cadena="Preprocessing valA corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

valCorpus="valA-"+L3code3+"-"+L2code3+".txt"
valPreCorpus="valA-pre-"+L3code3+"-"+L2code3+".txt"

entrada=codecs.open(valCorpus,"r",encoding="utf-8")
sortida=codecs.open(valPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_L3:
            toksl=tokenizerL3.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_L2:
            toktl=tokenizerL2.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_L1:
                toksl=truecaserL1.truecase(toksl)
            if TRUECASE_L3:
                toktl=truecaserL3.truecase(toktl)
            
            #if not weight==None:
            #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            #else:
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()


if VERBOSE:
    cadena="Preprocessing valB corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

valCorpus="valB-"+L1code3+"-"+L2code3+".txt"
valPreCorpus="valB-pre-"+L1code3+"-"+L2code3+".txt"

entrada=codecs.open(valCorpus,"r",encoding="utf-8")
sortida=codecs.open(valPreCorpus,"w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=None
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_L1:
            toksl=tokenizerL1.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_L2:
            toktl=tokenizerL2.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_L1:
                toksl=truecaserL1.truecase(toksl)
            if TRUECASE_L2:
                toktl=truecaserL2.truecase(toktl)
            
            #if not weight==None:
            #    cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            #else:
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())
            sortida.write(cadena+"\n")
    
entrada.close()
sortida.close()

#SENTENCEPIECE

if VERBOSE:
    cadena="Start of sentencepiece process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

    
if TRAIN_SENTENCEPIECE:
    if VERBOSE:
        cadena="Start of sentencepiece training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    #corpusL1
    outconcat=codecs.open("tempL1.temp","w",encoding="utf-8")
    trainCorpus="trainB-pre-"+L1code3+"-"+L2code3+".txt"
    entradaconcat=codecs.open(trainCorpus,"r",encoding="utf-8")
    for linia in entradaconcat:
        linia=linia.rstrip()
        camps=linia.split("\t")
        outconcat.write(camps[0]+"\n")
    entradaconcat.close()

    filename="corpusTOSP-"+L1code3+".txt"
    sortidatemp=codecs.open(filename,"w",encoding="utf-8")

    entradatemp=codecs.open("tempL1.temp","r",encoding="utf-8")

    cont=0
    for linia in entradatemp:
        linia=linia.rstrip()
        sortidatemp.write(linia+"\n")
        cont+=1
        if cont>SAMPLE_SIZE:
            break
    entradatemp.close()
    sortidatemp.close()

    #corpusL3
    outconcat=codecs.open("tempL3.temp","w",encoding="utf-8")
    trainCorpus="trainA-pre-"+L3code3+"-"+L2code3+".txt"
    entradaconcat=codecs.open(trainCorpus,"r",encoding="utf-8")
    for linia in entradaconcat:
        linia=linia.rstrip()
        camps=linia.split("\t")
        outconcat.write(camps[0]+"\n")
    entradaconcat.close()

    filename="corpusTOSP-"+L3code3+".txt"
    sortidatemp=codecs.open(filename,"w",encoding="utf-8")

    entradatemp=codecs.open("tempL3.temp","r",encoding="utf-8")

    cont=0
    for linia in entradatemp:
        linia=linia.rstrip()
        sortidatemp.write(linia+"\n")
        cont+=1
        if cont>SAMPLE_SIZE:
            break
        
    entradatemp.close()
    sortidatemp.close()

    #corpusL2
    outconcat=codecs.open("tempL2.temp","w",encoding="utf-8")
    
    trainCorpus="trainA-pre-"+L3code3+"-"+L2code3+".txt"
    entradaconcat=codecs.open(trainCorpus,"r",encoding="utf-8")
    for linia in entradaconcat:
        linia=linia.rstrip()
        camps=linia.split("\t")
        outconcat.write(camps[1]+"\n")
    
    trainCorpus="trainB-pre-"+L1code3+"-"+L2code3+".txt"
    entradaconcat=codecs.open(trainCorpus,"r",encoding="utf-8")
    for linia in entradaconcat:
        linia=linia.rstrip()
        camps=linia.split("\t")
        outconcat.write(camps[1]+"\n")
    entradaconcat.close()

    filename="corpusTOSP-"+L2code3+".txt"
    sortidatemp=codecs.open(filename,"w",encoding="utf-8")
    entradatemp=codecs.open("tempL2.temp","r",encoding="utf-8")

    cont=0
    for linia in entradatemp:
        linia=linia.rstrip()
        cont+=1
        if cont>SAMPLE_SIZE:
            break
        sortidatemp.write(linia+"\n")
    entradatemp.close()
    sortidatemp.close()
  
    sentencepiece_train("corpusTOSP-"+L1code3+".txt","corpusTOSP-"+L3code3+".txt","corpusTOSP-"+L2code3+".txt",L1code2,L3code2,L2code2,SP_MODEL_PREFIX,MODEL_TYPE,VOCAB_SIZE,CHARACTER_COVERAGE,INPUT_SENTENCE_SIZE,SPLIT_DIGITS,CONTROL_SYMBOLS,USER_DEFINED_SYMBOLS)




#ENCODING CORPORA WITH SENTENCEPIECE
if VERBOSE:
    cadena="Encoding corpora with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

SP_MODEL=SP_MODEL_PREFIX+".model"

#CORPUS A

trainPreCorpus="trainA-pre-"+L3code3+"-"+L2code3+".txt"
out1="trainA."+L3code2
out2="trainA."+L2code2

entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
sortida1=codecs.open(out1,"w",encoding="utf-8")
sortida2=codecs.open(out2,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    sortida1.write(camps[0]+"\n")
    sortida2.write(camps[1]+"\n")
entrada.close()
sortida1.close()
sortida2.close()
   

if VERBOSE:
    cadena="Encoding train A corpus "+L3code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="trainA."+L3code2
outfile="trainA.sp."+L3code2
vocabulary_file="vocab_file."+L3code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

if VERBOSE:
    cadena="Encoding train A corpus "+L2code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="trainA."+L2code2
outfile="trainA.sp."+L2code2
vocabulary_file="vocab_file."+L2code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

#valA
valPreCorpus="valA-pre-"+L3code3+"-"+L2code3+".txt"
out1="valA."+L3code2
out2="valA."+L2code2

entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
sortida1=codecs.open(out1,"w",encoding="utf-8")
sortida2=codecs.open(out2,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    sortida1.write(camps[0]+"\n")
    sortida2.write(camps[1]+"\n")
entrada.close()
sortida1.close()
sortida2.close()
   

if VERBOSE:
    cadena="Encoding val A corpus "+L3code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="valA."+L3code2
outfile="valA.sp."+L3code2
vocabulary_file="vocab_file."+L3code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

if VERBOSE:
    cadena="Encoding val A corpus "+L2code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="valA."+L2code2
outfile="valA.sp."+L2code2
vocabulary_file="vocab_file."+L2code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

#CORPUS B

trainPreCorpus="trainB-pre-"+L1code3+"-"+L2code3+".txt"
out1="trainB."+L1code2
out2="trainB."+L2code2

entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
sortida1=codecs.open(out1,"w",encoding="utf-8")
sortida2=codecs.open(out2,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    sortida1.write(camps[0]+"\n")
    sortida2.write(camps[1]+"\n")
entrada.close()
sortida1.close()
sortida2.close()
   

if VERBOSE:
    cadena="Encoding train B corpus "+L1code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="trainB."+L1code2
outfile="trainB.sp."+L1code2
vocabulary_file="vocab_file."+L1code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

if VERBOSE:
    cadena="Encoding train B corpus "+L2code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="trainB."+L2code2
outfile="trainB.sp."+L2code2
vocabulary_file="vocab_file."+L2code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

#valB
valPreCorpus="valB-pre-"+L1code3+"-"+L2code3+".txt"
out1="valB."+L1code2
out2="valB."+L2code2

entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
sortida1=codecs.open(out1,"w",encoding="utf-8")
sortida2=codecs.open(out2,"w",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    sortida1.write(camps[0]+"\n")
    sortida2.write(camps[1]+"\n")
entrada.close()
sortida1.close()
sortida2.close()
   

if VERBOSE:
    cadena="Encoding val B corpus "+L1code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="valB."+L1code2
outfile="valB.sp."+L1code2
vocabulary_file="vocab_file."+L1code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)

if VERBOSE:
    cadena="Encoding val B corpus "+L2code2+" with sentencepiece: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

infile="valB."+L2code2
outfile="valB.sp."+L2code2
vocabulary_file="vocab_file."+L2code2
sentencepiece_encode(infile,outfile, SP_MODEL,vocabulary_file,VOCABULARY_THRESHOLD,bosSP,eosSP)


nullweights=codecs.open("nullweights.temp","w",encoding="utf-8")
nullweights.close()

if GUIDED_ALIGNMENT:
    if VERBOSE:
        cadena="Guided alignment training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    if DELETE_EXISTING:
        FILE="trainA.sp."+L3code2+"."+L2code2+".align" 
        if os.path.exists(FILE):
            os.remove(FILE)
        FILE="trainB.sp."+L1code2+"."+L2code2+".align" 
        if os.path.exists(FILE):
            os.remove(FILE)
    if ALIGNER=="fast_align":
        if VERBOSE:
            cadena="Fast_align: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_fast_align(MTUOC,"trainA.sp","trainA.sp","nullweights.temp",L3code2,L2code2,False,VERBOSE)
        guided_alignment_fast_align(MTUOC,"trainB.sp","trainB.sp","nullweights.temp",L1code2,L2code2,False,VERBOSE)
        
    elif ALIGNER=="eflomal":
        if VERBOSE:
            cadena="Eflomal: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_eflomal(MTUOC,"trainA.sp","trainA.sp","nullweights.temp",L3code2,L2code2,SPLIT_LIMIT,VERBOSE)
        guided_alignment_eflomal(MTUOC,"trainB.sp","trainB.sp","nullweights.temp",L1code2,L2code2,SPLIT_LIMIT,VERBOSE)
 
if GUIDED_ALIGNMENT_VALID:
    if VERBOSE:
        cadena="Guided alignment valid: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    if DELETE_EXISTING:
        FILE="valA.sp."+L3code2+"."+L2code2+".align" 
        if os.path.exists(FILE):
            os.remove(FILE)
        FILE="valB.sp."+L1code2+"."+L2code2+".align" 
        if os.path.exists(FILE):
            os.remove(FILE)
    if ALIGNER=="fast_align":
        if VERBOSE:
            cadena="Fast_align: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_fast_align(MTUOC,"valA.sp","valA.sp","nullweights.temp",L3code2,L2code2,False,VERBOSE)
        guided_alignment_fast_align(MTUOC,"valB.sp","valB.sp","nullweights.temp",L1code2,L2code2,False,VERBOSE)
        
    elif ALIGNER=="eflomal":
        if VERBOSE:
            cadena="Eflomal: "+str(datetime.now())
            print(cadena)
            logfile.write(cadena+"\n")
        guided_alignment_eflomal(MTUOC,"valA.sp","valA.sp","nullweights.temp",L3code2,L2code2,SPLIT_LIMIT,VERBOSE)
        guided_alignment_eflomal(MTUOC,"valB.sp","valB.sp","nullweights.temp",L1code2,L2code2,SPLIT_LIMIT,VERBOSE)
if VERBOSE:
    cadena="End of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

#DELETE TEMPORAL FILES


if DELETE_TEMP:
    if VERBOSE:
        cadena="Deleting temporal files: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
    for f in glob.glob("*.temp"):
        os.remove(f)
