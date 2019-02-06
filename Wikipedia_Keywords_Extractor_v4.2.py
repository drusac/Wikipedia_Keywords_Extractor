import wikipedia
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

#pages_array = ['Alfa_Romeo', 'Aston_Martin', 'Audi', 'Automobile_Dacia', 'Bentley', 'BMW', 'Bugatti', 'Cadillac', 'Chevrolet', 'Citroën', 'Deutsche_Tourenwagen_Masters', 'Dodge', 'Ducati_Motor_Holding_S.p.A.', 'Electric_car', 'Fiat_Automobiles', 'Ford_Motor_Company', 'Formula_One', 'Honda', 'Hyundai', 'Jaguar_Cars', 'Jeep', 'Kia_Motors', 'Lada', 'Lamborghini', 'Lancia', 'Land_Rover', 'Lexus', 'Lincoln_Motor_Company', 'Lotus_Cars', 'Maserati', 'Mazda', 'McLaren_Automotive', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rimac_Automobili', 'Saab_Automobile', 'SEAT', 'Subaru', 'Suzuki', 'Tesla,_Inc.', 'Toyota', 'Volkswagen', 'Volvo', 'World_Rally_Championship', 'Zastava_Automobiles']
pages_array = ['Alfa_Romeo', 'Aston_Martin', 'Audi', 'Automobile_Dacia', 'Bentley']

print("\n")
keyword_number = int(input('Broj željenih najfrekventnijih ključnih riječi po dokumentu: '))

save_to_counter = 0

dictionary = {}


for i in pages_array:
    page = wikipedia.page(i)
    document = page.content
    
    add_text_extension = str(i) + ".txt"
    file = open("Generated_txt/" + add_text_extension, 'w', encoding = 'utf-8')
    file.write(document)
    file.close()

    
    file = open("Generated_txt/" + add_text_extension, "r", encoding = 'utf-8')
    contents = file.read()
    
    #recenice = sent_tokenize(contents)
    words = word_tokenize(contents)
    
    file.close()
    
    words_lowercase = [word.lower() for word in words if word.isalpha() or word.isnumeric()] # ili bez "or word.isnumeric() ako ne zelimo brojeve
    
    
    add_name_lemmatized = str(i) + "_Lemmatized.txt"
    file = open("Generated_txt/" + add_name_lemmatized, 'w', encoding = 'utf-8')
    
    
    x = 0
    for word in words_lowercase:
        lemmatized_word = lemmatizer.lemmatize(word, pos = "v")
    #    print (lemmatized_word)
        if lemmatized_word not in stop_words:
    #        print (lemmatized_word)
            file.write(lemmatized_word)
            file.write(", ")
            x += 1
            
    file.close()
  
        
    print("{0:30}{1:5}{2:7}".format(i, "Broj riječi:", x))

    
    file = open("Generated_txt/" + add_name_lemmatized, 'r', encoding = 'utf-8')
    contents = file.read()
    

    vectorizer = TfidfVectorizer(max_features = keyword_number) #keyword_number umjesto 'broj'
    response = vectorizer.fit_transform([contents])
    
    file.close()
    
    add_name_weights = str(i) + "_Weights.txt"
    file = open("Generated_txt/" + add_name_weights, 'w', encoding = 'utf-8')
    
    add_name_top_keywords = str(i) + "_Top_Keywords.txt"
    file = open("Generated_txt/" + add_name_top_keywords, 'w', encoding = 'utf-8')
    
    
#    // IMENA ZNAČAJKI ODNOSNO KLJUČNE RIJEČE
    feature_names = vectorizer.get_feature_names()

#    // ISPIS TEŽINA KLJUČNIH RIJEČI U CONSOLU
    for col in response.nonzero()[1]:
        print (feature_names[col], ' - ', response[0, col])

#    // ZASPISIVANJE TEŽINA KLJUČNIH RIJEČI U *_WEIGHTS DATOTEKU
    with open("Generated_txt/" + add_name_weights, 'w') as f:
        for col in response.nonzero()[1]:
            print(response[0, col], ' - ', feature_names[col], file = f)
            
#    // ZASPISIVANJE TEŽINA KLJUČNIH RIJEČI U *_WEIGHTS DATOTEKU
    with open("Generated_txt/" + add_name_top_keywords, 'w') as f:
        for col in response.nonzero()[1]:
            print(feature_names[col], file = f)
            
#    // ČITANJE TOP N KLJUČNIH RIJEČI IZ DATOTEKE TE ZAPISIVANJE SVIH N KLJUČNIH RIJEČI IZ POJEDINE DATOTEKE U POLJE
#    file = open("Generated_txt/" + add_name_top_keywords, 'r', encoding = 'utf-8')
    file = open("Generated_txt/" + add_name_top_keywords, 'r')
    top_n_keywords = file.read().splitlines()
#    // ISPIS POJEDINOG POLJA/DOKUMENTA (LISTE) KLJUČNIH RIJEČI
#    print('\nLista top n ključnih riječi u polju:', top_n_keywords)


    print("\n")


   
    dictionary["{}".format(i)]= top_n_keywords

    
    

    file.close()
    

print("======================================== REPORT ========================================\nPreuzet tekst pojedine stranice sa Wikipedije, odrađena lematizacija i izbacivanje stop riječi, izračunate težine za", keyword_number, "najfrekventnijih ključnih riječi. Generirano 4 .txt dokumenta za pojedinu Wikipedia stranicu.\n======================================== REPORT ========================================")
print("\n\n")
print("\n\n")
print("\n\n")

#counter_words_freq = 0

#counter = 0
# // WORK IN PROGRESS......
#for doc_number in range(0, 4):
#    counter = 0
#    for word in range(0, 20):
#        for word2 in range(0, 20):
#            for doc_iter in range(0, 4):
#                if (dictionary[pages_array[doc_number]][word] == dictionary[pages_array[doc_iter+1]][word2]):
#                    print(dictionary[pages_array[doc_number]][word])
#                    counter += 1
#    print("\n\n", counter)
         
# // WORK IN PROGRESS...... 
    
    
    
    