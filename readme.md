## Deney 1 — Baseline CNN

### Amaç

BloodMNIST üzerinde, simetri tabanlı veri artırma veya grup eşdeğişimli katman kullanmayan temel bir CNN eğitildi. Bu deney diğer yöntemlerle karşılaştırma için başlangıç noktasıdır.

### Veri ve protokol

MedMNIST’in resmi train, validation ve test bölünmeleri kullanıldı. Model eğitimi ve checkpoint seçimi sırasında test kümesi kullanılmadı.

| Bölünme | Görüntü sayısı |
|---|---:|
| Train | 11.959 |
| Validation | 1.712 |
| Test | 3.421 |

Görüntüler 3 kanallı ve 28×28 pikseldir. Etiketler sekiz kan hücresi sınıfından birini belirtir. Eğitim ve validation görüntülerine yalnızca `ToTensor()` uygulandı.

| Ayar | Değer |
|---|---|
| Random seed | 42 |
| Batch size | 32 |
| Epoch sayısı | 20 |
| Optimizer | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Loss | CrossEntropyLoss |
| Model seçimi | Validation macro-F1 |
| Eğitilebilir parametre sayısı | 421.960 |

CNN iki convolution katmanı, ReLU aktivasyonları, max pooling katmanları ve tam bağlı bir sınıflandırıcıdan oluşur. Model sıfırdan eğitilmiştir.

### Validation sonuçları

En iyi checkpoint, **19. epoch’ta** validation macro-F1’e göre seçildi.

| Metrik | Sonuç |
|---|---:|
| Accuracy | 0,9258 |
| Macro-precision | 0,9231 |
| Macro-recall | 0,9129 |
| Macro-F1 | 0,9175 |
| Weighted F1 | 0,9266 |

### Sınıf bazında sonuçlar

| Sınıf | Precision | Recall | F1-score | Örnek |
|---|---:|---:|---:|---:|
| Basophil | 0,8750 | 0,8607 | 0,8678 | 122 |
| Eosinophil | 0,9748 | 0,9904 | 0,9825 | 312 |
| Erythroblast | 0,9467 | 0,9161 | 0,9311 | 155 |
| Immature granulocytes | 0,7875 | 0,8690 | 0,8262 | 290 |
| Lymphocyte | 0,9402 | 0,9016 | 0,9205 | 122 |
| Monocyte | 0,8741 | 0,8252 | 0,8489 | 143 |
| Neutrophil | 0,9906 | 0,9489 | 0,9693 | 333 |
| Platelet | 0,9957 | 0,9915 | 0,9936 | 235 |

### Gözlemler

Eğitim ilerledikçe train ve validation loss azaldı; accuracy ve macro-F1 yükseldi. Eğrilerde belirgin ve kalıcı bir train-validation ayrışması görülmedi. Bu koşuda 20 epoch sonunda açık bir overfitting işareti gözlenmedi.

En yüksek sınıf F1 değeri platelet ve eosinophil sınıflarında elde edildi. En düşük sınıf precision değeri immature granulocytes sınıfında görüldü. Confusion matrix’te öne çıkan karışmalar:

- 22 monocyte örneği immature granulocytes olarak tahmin edildi.
- 16 immature granulocytes örneği monocyte olarak tahmin edildi.
- 14 basophil örneği immature granulocytes olarak tahmin edildi.
- 12 neutrophil örneği immature granulocytes olarak tahmin edildi.

### Yorum ve sınırlılıklar

Baseline, validation kümesinde 0,9175 macro-F1 elde etti. Bu sonuç yalnızca validation performansıdır; nihai test performansı olarak yorumlanmamalıdır. Model seçimi için validation kümesi kullanıldı. Test kümesi, üç yaklaşım tamamlandıktan sonra yalnızca final değerlendirmede kullanılacaktır.

Eğitim süresi bu rapora eklenen konsol çıktısında yer almadığından raporlanmamıştır. Eğitim süresi karşılaştırma tablosuna ekleneceği zaman eğitim çıktısından ayrıca kaydedilmelidir.

### Grafikler

**Eğitim ve validation eğrileri**

![Baseline CNN eğitim ve validation eğrileri](outputs/plots/baseline_training_curves.png)

**Validation confusion matrix**

![Baseline CNN validation confusion matrix](outputs/plots/baseline_val_confusion_matrix.png)

## Deney 2 — P4 Simetri Veri Artırmalı CNN

### Amaç

Bu deneyde baseline CNN’in eğitim görüntülerine rastgele 0°, 90°, 180° veya 270° dönüş uygulandı. Model mimarisi ve eğitim ayarları baseline ile aynı tutuldu. Validation kümesine rastgele eğitim augmentasyonu uygulanmadı.

P4 dönüşleri `torch.rot90` ile uygulandı. Böylece pikseller ara açılarla döndürülürken yeniden örneklenmedi. Kan hücresinin kadraj içindeki yönünün etiket bilgisini değiştirmediği varsayıldı.

### Deney ayarları

| Ayar | Değer |
|---|---|
| Random seed | 42 |
| Batch size | 32 |
| Epoch sayısı | 20 |
| Optimizer | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Loss | CrossEntropyLoss |
| Model seçimi | Validation macro-F1 |
| Eğitilebilir parametre sayısı | 421.960 |

### Validation sonuçları

Aşağıdaki metrikler paylaşılan validation confusion matrix’inden hesaplanmıştır ve dört ondalık basamağa yuvarlanmıştır.

| Metrik | Sonuç |
|---|---:|
| Accuracy | 0,9252 |
| Macro-precision | 0,9259 |
| Macro-recall | 0,9107 |
| Macro-F1 | 0,9174 |

### Sınıf bazında sonuçlar

Değerler paylaşılan confusion matrix’inden hesaplanmıştır.

| Sınıf | Precision | Recall | F1-score | Örnek |
|---|---:|---:|---:|---:|
| Basophil | 0,9123 | 0,8525 | 0,8814 | 122 |
| Eosinophil | 0,9903 | 0,9808 | 0,9855 | 312 |
| Erythroblast | 0,9662 | 0,9226 | 0,9439 | 155 |
| Immature granulocytes | 0,7731 | 0,8931 | 0,8288 | 290 |
| Lymphocyte | 0,9397 | 0,8934 | 0,9160 | 122 |
| Monocyte | 0,8456 | 0,8042 | 0,8244 | 143 |
| Neutrophil | 0,9843 | 0,9429 | 0,9632 | 333 |
| Platelet | 0,9957 | 0,9957 | 0,9957 | 235 |

### Grafikler

**Eğitim ve validation eğrileri**

![P4 augmentasyonlu CNN eğitim ve validation eğrileri](outputs/plots/p4_augmented_training_curves.png)

**Validation confusion matrix**

![P4 augmentasyonlu CNN validation confusion matrix](outputs/plots/p4_augmented_val_confusion_matrix.png)

### Baseline ile karşılaştırma ve gözlemler

Baseline validation accuracy değeri 0,9258, macro-F1 değeri 0,9175 idi. P4 augmentasyonlu CNN’de accuracy yaklaşık 0,9252, macro-F1 yaklaşık 0,9174 oldu. Bu tek koşuda genel validation performansı baseline’dan belirgin biçimde farklılaşmadı.

Sınıf bazında değişimler aynı yönde olmadı. Basophil ve erythroblast F1 değerleri yükselirken monocyte F1 değeri düştü. Augmentasyonlu confusion matrix’te en sık karışan çiftler arasında 25 monocyte örneğinin immature granulocytes tahmin edilmesi ve 14’er basophil ve neutrophil örneğinin immature granulocytes tahmin edilmesi yer aldı.

Eğitim eğrilerinde loss düşmeye devam etti ve belirgin bir train-validation ayrışması görülmedi. Eğitim sırasında rastgele dönüşüm ve dropout kullanılması, train metriklerini validation metriklerinden düşük tutabilir; bu fark tek başına daha iyi genelleme kanıtı değildir.

Bu sonuçlar validation kümesine aittir. Test kümesi bu deneyde kullanılmamıştır.