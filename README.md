# NLP-tessearct-Text-Recognition
Here is the code to convert an Image file or a PDF file into text and tokenize it according to the requirements.

```python
import cv2
import pytesseract

img = cv2.imread('file.png')

config = ('-l eng --oem 1 --psm 3')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
ret, threshimg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
  
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
  
dilation = cv2.dilate(threshimg, rect_kernel, iterations = 1) 
  
img_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                 cv2.CHAIN_APPROX_NONE) 
  
for cnt in img_contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    cropped_img = img[y:y + h, x:x + w] 
      
    file = open("recognized.txt", "a") 
      
    text = pytesseract.image_to_string(cropped_img) 
      
    file.write(text) 
    print(text)
    file.write("\n") 
      
    file.close 
```

    macro
    weighted
    
    
    
    
    accuracy
    
    
    690
    690
    690
    
    
    0.64
    0.64
    0.64
    
    
    
    0.49
    0.51
    0.59
    0.89
    0.67
    0.70
    0.70
    0.74
    0.34
    0.98
    0.77
    0.44
    0.29
    0.51
    0.79
    0.67
    0.74
    0.54
    0.62
    0.56
    0.75
    0.62
    0.77
    
    
    0.46
    0.50
    0.67
    0.93
    0.72
    0.66
    0.73
    0.81
    0.30
    1.00
    0.76
    0.47
    0.32
    0.53
    0.83
    0.71
    0.81
    0.50
    0.67
    0.50
    0.76
    0.54
    0.67
    
    
    oooooocooocoocoocoocooocooocooeoocoecocoeoeocece
    
    53
    52
    53
    -86
    62
    -76
    -68
    -68
    41
    96
    -78
    42
    27
    49
    76
    -63
    -68
    59
    57
    64
    73
    -73
    92
    
    
    apple pie
    baklava
    
    burger
    
    edamame
    eggs_benedict
    fried_rice
    frozen_yogurt
    hot_dog
    huevos_rancheros
    miso_soup
    mussels
    
    nachos
    
    omelette
    pork_chop
    poutine
    
    samosa
    
    sashimi
    
    scallops
    shrimp_and_grits
    spaghetti_bolognese
    spaghetti_carbonara
    spring rolls
    waffles
    
    
    support
    
    
    f1-score
    
    
    recall
    
    
    precision
    
    
    
    Report:
    
    
    Classification
    
    



```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
```

    [nltk_data] Downloading package punkt to /Users/rafay/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    True




```python
f = open("recognized.txt", "r")
f = f.read()
```


```python
f = f.replace("\n",' ')
```


```python
f = f.replace("0.",'')
```


```python
f = word_tokenize(f)
```


```python
check_list = ['(',')','.',':','""',',',"''"]
```


```python
for i in check_list:
    if i in f:
        f.remove(i)
```


```python
print(f)
```

    ['dept', 'name', 'of', 'organization', 'of', 'Aff', 'name', 'of', 'organization', 'of', 'Aff', 'City', 'Country', 'email', 'address', 'or', 'ORCID', '6', 'Given', 'Name', 'Surname', 'macro', 'weighted', 'accuracy', '690', '690', '690', '64', '64', '64', '49', '51', '59', '89', '67', '70', '70', '74', '34', '98', '77', '44', '29', '51', '79', '67', '74', '54', '62', '56', '75', '62', '77', '46', '50', '67', '93', '72', '66', '73', '81', '30', '1.00', '76', '47', '32', '53', '83', '71', '81', '50', '67', '50', '76', '54', '67', 'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece', '53', '52', '53', '-86', '62', '-76', '-68', '-68', '41', '96', '-78', '42', '27', '49', '76', '-63', '-68', '59', '57', '64', '73', '-73', '92', 'apple', 'pie', 'baklava', 'burger', 'edamame', 'eggs_benedict', 'fried_rice', 'frozen_yogurt', 'hot_dog', 'huevos_rancheros', 'miso_soup', 'mussels', 'nachos', 'omelette', 'pork_chop', 'poutine', 'samosa', 'sashimi', 'scallops', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring', 'rolls', 'waffles', 'support', 'f1-score', 'recall', 'precision', 'Report', 'Classification', 'macro', 'weighted', 'accuracy', '690', '690', '690', '64', '64', '64', '49', '51', '59', '89', '67', '70', '70', '74', '34', '98', '77', '44', '29', '51', '79', '67', '74', '54', '62', '56', '75', '62', '77', '46', '50', '67', '93', '72', '66', '73', '81', '30', '1.00', '76', '47', '32', '53', '83', '71', '81', '50', '67', '50', '76', '54', '67', 'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece', '53', '52', '53', '-86', '62', '-76', '-68', '-68', '41', '96', '-78', '42', '27', '49', '76', '-63', '-68', '59', '57', '64', '73', '-73', '92', 'apple', 'pie', 'baklava', 'burger', 'edamame', 'eggs_benedict', 'fried_rice', 'frozen_yogurt', 'hot_dog', 'huevos_rancheros', 'miso_soup', 'mussels', 'nachos', 'omelette', 'pork_chop', 'poutine', 'samosa', 'sashimi', 'scallops', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring', 'rolls', 'waffles', 'support', 'f1-score', 'recall', 'precision', 'Report', ':', 'Classification', 'macro', 'weighted', 'accuracy', '690', '690', '690', '64', '64', '64', '49', '51', '59', '89', '67', '70', '70', '74', '34', '98', '77', '44', '29', '51', '79', '67', '74', '54', '62', '56', '75', '62', '77', '46', '50', '67', '93', '72', '66', '73', '81', '30', '1.00', '76', '47', '32', '53', '83', '71', '81', '50', '67', '50', '76', '54', '67', 'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece', '53', '52', '53', '-86', '62', '-76', '-68', '-68', '41', '96', '-78', '42', '27', '49', '76', '-63', '-68', '59', '57', '64', '73', '-73', '92', 'apple', 'pie', 'baklava', 'burger', 'edamame', 'eggs_benedict', 'fried_rice', 'frozen_yogurt', 'hot_dog', 'huevos_rancheros', 'miso_soup', 'mussels', 'nachos', 'omelette', 'pork_chop', 'poutine', 'samosa', 'sashimi', 'scallops', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring', 'rolls', 'waffles', 'support', 'f1-score', 'recall', 'precision', 'Report', ':', 'Classification', 'macro', 'weighted', 'accuracy', '690', '690', '690', '64', '64', '64', '49', '51', '59', '89', '67', '70', '70', '74', '34', '98', '77', '44', '29', '51', '79', '67', '74', '54', '62', '56', '75', '62', '77', '46', '50', '67', '93', '72', '66', '73', '81', '30', '1.00', '76', '47', '32', '53', '83', '71', '81', '50', '67', '50', '76', '54', '67', 'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece', '53', '52', '53', '-86', '62', '-76', '-68', '-68', '41', '96', '-78', '42', '27', '49', '76', '-63', '-68', '59', '57', '64', '73', '-73', '92', 'apple', 'pie', 'baklava', 'burger', 'edamame', 'eggs_benedict', 'fried_rice', 'frozen_yogurt', 'hot_dog', 'huevos_rancheros', 'miso_soup', 'mussels', 'nachos', 'omelette', 'pork_chop', 'poutine', 'samosa', 'sashimi', 'scallops', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring', 'rolls', 'waffles', 'support', 'f1-score', 'recall', 'precision', 'Report', ':']



```python
temp = 0
for i in f:
    if i.isnumeric():
        temp = temp + int(i)
        f.remove(i)
```


```python
temp
```




    13938




```python

f
```




    ['dept',
     'name',
     'of',
     'organization',
     'of',
     'Aff',
     'name',
     'of',
     'organization',
     'of',
     'Aff',
     'City',
     'Country',
     'email',
     'address',
     'or',
     'ORCID',
     'Given',
     'Name',
     'Surname',
     'macro',
     'weighted',
     'accuracy',
     '51',
     '89',
     '98',
     '44',
     '51',
     '54',
     '56',
     '46',
     '72',
     '73',
     '30',
     '1.00',
     '47',
     '71',
     '54',
     'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece',
     '52',
     '-86',
     '-76',
     '-68',
     '-68',
     '96',
     '-78',
     '27',
     '-63',
     '-68',
     '57',
     '73',
     '-73',
     'apple',
     'pie',
     'baklava',
     'burger',
     'edamame',
     'eggs_benedict',
     'fried_rice',
     'frozen_yogurt',
     'hot_dog',
     'huevos_rancheros',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'pork_chop',
     'poutine',
     'samosa',
     'sashimi',
     'scallops',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring',
     'rolls',
     'waffles',
     'support',
     'f1-score',
     'recall',
     'precision',
     'Report',
     'Classification',
     'macro',
     'weighted',
     'accuracy',
     '51',
     '89',
     '98',
     '44',
     '51',
     '54',
     '56',
     '46',
     '72',
     '73',
     '30',
     '1.00',
     '47',
     '71',
     '50',
     '50',
     '54',
     'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece',
     '52',
     '-86',
     '-76',
     '-68',
     '-68',
     '96',
     '-78',
     '27',
     '-63',
     '-68',
     '57',
     '73',
     '-73',
     'apple',
     'pie',
     'baklava',
     'burger',
     'edamame',
     'eggs_benedict',
     'fried_rice',
     'frozen_yogurt',
     'hot_dog',
     'huevos_rancheros',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'pork_chop',
     'poutine',
     'samosa',
     'sashimi',
     'scallops',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring',
     'rolls',
     'waffles',
     'support',
     'f1-score',
     'recall',
     'precision',
     'Report',
     ':',
     'Classification',
     'macro',
     'weighted',
     'accuracy',
     '690',
     '64',
     '64',
     '64',
     '51',
     '89',
     '70',
     '70',
     '74',
     '98',
     '44',
     '51',
     '74',
     '54',
     '56',
     '46',
     '50',
     '67',
     '72',
     '73',
     '30',
     '1.00',
     '47',
     '71',
     '50',
     '67',
     '50',
     '54',
     '67',
     'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece',
     '52',
     '53',
     '-86',
     '62',
     '-76',
     '-68',
     '-68',
     '96',
     '-78',
     '27',
     '76',
     '-63',
     '-68',
     '57',
     '64',
     '73',
     '-73',
     'apple',
     'pie',
     'baklava',
     'burger',
     'edamame',
     'eggs_benedict',
     'fried_rice',
     'frozen_yogurt',
     'hot_dog',
     'huevos_rancheros',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'pork_chop',
     'poutine',
     'samosa',
     'sashimi',
     'scallops',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring',
     'rolls',
     'waffles',
     'support',
     'f1-score',
     'recall',
     'precision',
     'Report',
     ':',
     'Classification',
     'macro',
     'weighted',
     'accuracy',
     '690',
     '690',
     '690',
     '64',
     '64',
     '64',
     '51',
     '89',
     '67',
     '70',
     '70',
     '74',
     '98',
     '44',
     '51',
     '67',
     '74',
     '54',
     '62',
     '56',
     '62',
     '46',
     '50',
     '67',
     '72',
     '73',
     '30',
     '1.00',
     '76',
     '47',
     '53',
     '71',
     '50',
     '67',
     '50',
     '76',
     '54',
     '67',
     'oooooocooocoocoocoocooocooocooeoocoecocoeoeocece',
     '53',
     '52',
     '53',
     '-86',
     '62',
     '-76',
     '-68',
     '-68',
     '96',
     '-78',
     '27',
     '76',
     '-63',
     '-68',
     '57',
     '64',
     '73',
     '-73',
     'apple',
     'pie',
     'baklava',
     'burger',
     'edamame',
     'eggs_benedict',
     'fried_rice',
     'frozen_yogurt',
     'hot_dog',
     'huevos_rancheros',
     'miso_soup',
     'mussels',
     'nachos',
     'omelette',
     'pork_chop',
     'poutine',
     'samosa',
     'sashimi',
     'scallops',
     'shrimp_and_grits',
     'spaghetti_bolognese',
     'spaghetti_carbonara',
     'spring',
     'rolls',
     'waffles',
     'support',
     'f1-score',
     'recall',
     'precision',
     'Report',
     ':']




```python
import PyPDF2
pdfFileObj = open('file2.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)
pageObj = pdfReader.getPage(0)
print(pageObj.extractText())
pdfFileObj.close()
```

    1
    Electronically verified report. No signature required. Lab reports should be interpreted by a physician in correlation with clinical and radiologic findings.Department Of Microbiology Urine Examination
    Physical Examination
    Colour: Yellow Turbidity/Deposit: +Reporting Time: 16-Jul-2022 00:38
    Chemical Examination
    Parameter Name Result Reference Value
    SP Gravity 1.005-1.030 1.030
    pH 5.0-8.0 5.0
    Leukocyte Estrases Nil Nil
    Nitrite Negative Nil
    Proteins Nil NilParameter Name Result Reference Value
    Sugar Nil Nil
    Ketones Nil Nil
    Urobilinogen Normal Normal
    Bilirubin Nil Nil
    Heamoglobin Nil Nil
    Crystals: Nil Casts: Nil Nil Nil /LPF /HPFMicroscopic Examination
    Parameter Name Result Reference Value Unit
    Pus Cells 0-5 0-5 /HPF
    Red Blood Cell 0-5 Nil /HPF
    Epithelial Cells 0-5 6-10 /HPF
    Amorphous Nil Nil /HPFParameter Name Result Reference Value Unit
    Organisms Nil Nil /HPF
    Yeast Cells Nil Nil /HPF
    Dead Sperms Nil Nil /HPF
    Misc Nil Nil /HPF
    This test was performed on automated Urinalysis system.Note:
    Patient Detail :
    Mrs.Momina .  47 (Y) / Female
    Mobile:
    03218474693 ,
    Registration Location:
    Lahore:DHA DD Phase-4 Chughtai Medical CenterRegistration Date:
    15-Jul-2022 21:39
    Reference:
    Standard.
    Consultant:
    . 72344-15-07 Case Number:89001-16-911503557 Patient Number:XXX
    Dr.Naghmana Mazhar
    Consultant Pathologist03111456789
     www.chughtailab.com
     07 Jail Road Lahore info@chughtailab.comDr. Qamar Sultana
    M.B.B.S., M.Phil
    Consultant MicrobiologistDr. Irim Iftikhar
    M.B.B.S., F.C.P.S.
    Consultant MicrobiologistProf Waheed UZ Tariq
    M.B.B.S., PhD. DpBact
    F.C.P.S, F.R.C.Path, F.R.C.P.EConsultant
    VirologistDr. Omar Chughtai
    M.B.B.S., M.D., F.C.A.P.
    Diplomate American Board of Anatomic
    and Clinical PathologyDr. A . S. Chughtai
    M.B.B.S., M.I.A.C., M.Phil.
    F.C.P.S., F.C.P.P.Consultant Pathologist



```python

```
