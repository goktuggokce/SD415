## 📌 Proje Hakkında

Bu proje, bireylerin kalp krizine yatkınlığını belirlemek amacıyla veri madenciliği tekniklerini kullanmaktadır. Mevcut veri seti, 13 farklı öznitelikten oluşmakta ve bireylerin sağlık durumlarıyla ilgili önemli veriler içermektedir. Projenin temel hedefi, bu verilerden yola çıkarak bireyleri "riskli" ve "risksiz" olarak sınıflandırmaktır. 🎯

Kalp krizi riski, bireylerin yaş, cinsiyet, kan basıncı, kolesterol seviyesi gibi özniteliklerle ilişkilendirilmiştir. Bu proje, KNN (K-Nearest Neighbors) algoritmasını kullanarak bu özniteliklerden anlamlı sonuçlar çıkarmayı hedeflemektedir. Ayrıca, veri görselleştirme ve analiz yöntemleri ile veri setindeki ilişkileri ve desenleri ortaya koymayı amaçlamaktadır. 🌟

---

## 🎯 Projenin Amacı

Projenin temel amacı, bireylerin test sonuçları ve kişisel sağlık verileri kullanılarak kalp krizi riskinin analiz edilmesidir. Veri madenciliği yöntemi olan KNN ile bu verilerin anlamlı hale getirilmesi ve sınıflandırma doğruluğunun artırılması hedeflenmiştir. 🚀 

Bunun yanı sıra, projede şu hedefler de bulunmaktadır:
- Büyük veri kümeleriyle çalışabilme ve veri madenciliği becerilerinin geliştirilmesi,
- Sınıflandırma algoritmaları hakkında pratik bilgi kazanılması,
- Sağlık sektörü gibi kritik alanlarda veri madenciliği uygulamalarının potansiyelini gösterme,
- Risk gruplarını tespit ederek bu bilgileri sağlık uzmanlarına fayda sağlayacak bir model haline getirme.

---

## 🧑🏻‍💻 Proje Ekibi
Bu proje, Afyon Kocatepe Üniversitesi Bilgisayar Mühendisliği Bölümü Veri Madenciliği Dersi kapsamında aşağıdaki ekip üyeleri tarafından geliştirilmiştir:

-  Mehmet Göktuğ Gökçe (212923025)
-  Sinan Malak (212923008)
-  Onur Barbaros (212923036)

---

## 📂 Veri Seti: Heart Attacks

Projede kullanılan veri seti, kalp krizi riskiyle ilgili bilgileri içermekte ve aşağıdaki detaylara sahiptir:

- **Kayıt Sayısı:** 303 
- **Sütun Sayısı:** 14 (13 öznitelik ve 1 hedef değişken)
- **Hedef Değişken:** `output` 
  - **1:** Riskli birey
  - **0:** Risksiz birey
- **Veri Seti Kaynağı:** Veri seti açık kaynak bir platform olan Kaggle üzerinden alınmıştır ve sağlık verileri analizinde sıklıkla kullanılmaktadır.

Bu veri seti, yaş, cinsiyet, kan şekeri, kolesterol seviyesi gibi bireylerin sağlık durumlarına dair önemli bilgileri içermektedir. Verilerin eksiksiz ve anlamlı şekilde işlenmesi, projenin başarıyla sonuçlanması açısından büyük önem taşımaktadır.

---

# 📊 Python Veri Analizi ve Modelleme

Bu proje, veri analizi ve veri madenciliği için gerekli temel kütüphanelerin kullanımı ile başlar. Proje kapsamında kullanılan kütüphaneler aşağıda belirtilmiştir.

---

## 📦 Gerekli Kütüphaneler

Proje boyunca aşağıdaki kütüphaneler kullanılmaktadır. Eğer bu kütüphaneler yüklü değilse, `pip install <kütüphane_adı>` komutunu kullanarak yükleyebilirsiniz.

### 📚 Projede Kullanılan Kütüphaneler

Projede kullanılan temel Python kütüphaneleri ve bunların işlevleri aşağıda açıklanmıştır:

Sayısal işlemler ve dizilerle çalışmak için

* import numpy as np

Model performans metrikleri için

* from sklearn.metrics import classification_report, accuracy_score  

Veri setini eğitim ve test setlerine ayırmak için

* from sklearn.model_selection import train_test_split

Verileri ölçeklendirmek için

* from sklearn.preprocessing import StandardScaler  

Grafik oluşturmak için

* import matplotlib.pyplot as plt  

Boyut indirgeme işlemleri için

* from sklearn.decomposition import PCA  

Renk haritası oluşturmak için

* from matplotlib.colors import ListedColormap  

Veri görselleştirme ve dağılım analizleri için

* import seaborn as sns  

Veriyi yükleme ve hazırlama

* import pandas as pd

