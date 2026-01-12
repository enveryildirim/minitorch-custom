 **Autodifferentiation (Otomatik Türev Alma)** motorunun temel yapı taşıdır.

Bu sınıflar, bir yazılım mühendisi olarak bildiğin standart fonksiyonlardan farklıdır. Standart bir fonksiyon sadece çıktı üretir (`return`). Ancak bu sınıflar, **hem ileriye doğru hesaplama yapmayı (forward) hem de geriye doğru hatayı yaymayı (backward)** bilen "çift yönlü" kapılardır.

Gel bu yapıyı 'Problem → Sezgi → Çözüm' çerçevesinde analiz edelim.

---

### 1. Problem: "Unutkan" Fonksiyonlar ve Hafıza İhtiyacı

Standart bir Python fonksiyonunu düşün:

```python
def carp(a, b):
    return a * b

```

Bu fonksiyon `a` ve `b`'yi çarpar ve sonucu döndürür. İşlem bittikten sonra `a` ve `b`'nin ne olduğunu unutur.

**Sorun:** Backpropagation (Geri Yayılım) sırasında, türevi hesaplamak için genellikle girdilerin orijinal değerlerine ihtiyacımız vardır.
Örneğin,  fonksiyonunun 'ya göre türevi 'dir. Eğer fonksiyon 'nin ne olduğunu unutursa, türevi hesaplayamazsınız.

**Çözüm:** Fonksiyonlarımız, hesaplama sırasında girdileri bir "zula"ya (Context) kaydetmeli ve geri yayılım sırasında bu zulayı kullanmalıdır.

---

### 2. Mimari Analiz: `ScalarFunction`

`ScalarFunction` sınıfı, bu "çift yönlü" ve "hafızalı" yapıyı kuran şablondur.

#### A. `forward(ctx, ...)`

* **Görevi:** Matematiksel işlemi yapar (örn. toplama, çarpma).
* **Kritik Fark:** `ctx` (Context) nesnesini parametre olarak alır. Gerekirse `ctx.save_for_backward(...)` ile girdileri hafızaya atar.

#### B. `backward(ctx, d_output)`

* **Görevi:** Zincir kuralını (Chain Rule) uygular.
* **Girdi (`d_output`):** Bir sonraki katmandan gelen gradyan ().
* **İşlem:** `ctx` içindeki saklanmış değerleri geri çağırır ve yerel türevle çarpar.
* **Çıktı:** Girdilere göre gradyanlar ().

#### C. `apply(...)`

* **Görevi:** Bu "sihirli" metot, kullanıcı kodunu (normal float değerler veya Scalar objeleri) hesaplama grafiğine bağlar.
* **Akış:**
1. Girdileri alır.
2. Bir `Context` (hafıza kutusu) oluşturur.
3. `forward` işlemini çalıştırır.
4. Sonucu ve oluşturulan geçmişi (`ScalarHistory`) içeren yeni bir `Scalar` nesresi döndürür. Bu, grafiği (DAG) oluşturan adımdır.



---

### 3. Subclass'ların Analizi ve Task 1.4 İçin Yol Haritası

Senin kodunda `forward` metodları Task 1.2 için kısmen doldurulmuş, ancak `backward` metodları (Task 1.4) eksik. Gel en önemlilerini derinlemesine inceleyelim.

#### 1. Çarpma (`Mul`)

* **Forward:** Zaten yapılmış. `ctx.save_for_backward(a, b)` kritik çünkü türev için ikisine de ihtiyaç var.
* **Backward Mantığı:**
* 'ya göre türev: 
* 'ye göre türev: 
* Zincir Kuralı: , 



```python
# Task 1.4 İpucu:
a, b = ctx.saved_values
return b * d_output, a * d_output

```

#### 2. Tersini Alma (`Inv`)

* **Matematik:** Türev 
* **Backward Mantığı:**
* Forward kısmında `a` kaydedilmiş.
* Zincir Kuralı: 



#### 3. Sigmoid (`Sigmoid`)

Derin öğrenmede çok kritiktir. Çıktıyı  aralığına sıkıştırır.


* **Mühendislik Hilesi:** Sigmoid'in türevi, kendi cinsinden ifade edilebilir:


* **Forward Analizi:** Kodda `ctx.save_for_backward(s)` satırı var. Dikkat et, girdi olan `a`'yı değil, çıktı olan `s`'yi kaydetmişiz. Neden?
* Çünkü türev formülü  yani .
* `a`'yı saklasaydık, türevde tekrar `exp` hesaplamak zorunda kalırdık (pahalı işlem). Sonucu saklamak performansı artırır.



```python
# Task 1.4 İpucu:
s = ctx.saved_values[0]
return d_output * (s * (1.0 - s))

```

#### 4. ReLU (`ReLU`)

* **Sezgi:** Negatifse öldür (0), pozitifse olduğu gibi geçir.
* **Türev:**
*  ise türev 1.
*  ise türev 0.


* **Backward Mantığı:** `save_for_backward(a)` yapılmış. `a`'yı kontrol et, pozitifse `d_output` aynen geçer, negatifse 0 döner.

#### 5. Karşılaştırma Operatörleri (`LT`, `EQ`)

* **Problem:** Bunlar "Step Function"dır (Basamak Fonksiyonu).
* **Türev:** Bu fonksiyonların türevi her yerde **0**'dır (Süreksizlik noktaları hariç, orada tanımsızdır ama bilgisayarda 0 kabul ederiz).
* **Neden?** 'yı çok az () değiştirirseniz,  durumu (True/False) değişmez. Değişim hızı 0'dır.
* **Backward:** `return 0.0, 0.0` (Bu türevlerin gradyan akışını kestiği anlamına gelir, bu yüzden bu işlemler genellikle backpropagation yolunda kullanılmaz).

---

### 4. Code Implementation (Örnek: Mul Class)

Task 1.4 için `Mul` sınıfını nasıl tamamlaman gerektiğini profesyonelce gösterelim:

```python
class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # Değerleri, backward adımında kullanmak üzere hafızaya alıyoruz.
        ctx.save_for_backward(a, b) 
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Hafızadan değerleri geri çağırıyoruz.
        # Python'un 'unpacking' özelliğini kullanıyoruz.
        a, b = ctx.saved_values
        
        # Zincir kuralı (Chain Rule):
        # dL/da = dL/dout * dout/da
        # dout/da = b
        grad_a = d_output * b
        
        # dL/db = dL/dout * dout/db
        # dout/db = a
        grad_b = d_output * a
        
        return grad_a, grad_b

```

---

### 5. Kritik Düşünme Meydan Okuması (Socratic Challenge)

> **Soru:** `Inv` (Ters alma) sınıfında forward metodunda `a` değerini saklıyoruz (`ctx.save_for_backward(a)`). Backward aşamasında türev  olarak hesaplanıyor.
> Eğer forward aşamasında `result = 1.0 / a` hesaplayıp, `a` yerine `result` değerini saklasaydık (`ctx.save_for_backward(result)`), backward fonksiyonunu nasıl daha **performanslı** (daha az bölme işlemi yaparak) yazardık?

**Cevabı Düşün:**
Türevi  cinsinden değil de sonuç () cinsinden yazarsan:
 olur.
Bu durumda `backward` içinde bölme işlemi yapmana gerek kalmazdı (çarpma bölmeden daha hızlıdır). Tıpkı `Sigmoid` örneğindeki optimizasyon gibi!

