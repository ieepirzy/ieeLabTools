# Dokumentaatio

Saat moduulin käyttöösi suorittamalla komennon:

```CLI
pip install ieeLabTools
```

<a href="https://pypi.org/project/ieeLabTools/">Moduuli on julkaistu PyPi alustalle</a>

## Pikaesimerkki:
```python
from ieeLabTools.core import Yvel
import sympy as sp
import numpy as np

#Luodaan esimerkkimuuttujat, näiden täytyy olla sympy-symboleja
U, I = sp.symbols("U I")
R = U/I #Rakennetaan symboleista/muututjista esimerkkifunktio
calc = Yvel(R) # Tässä luodaan Yvel-objekti yllä määritetyllä funktiolla. 
#TS. Jotta voit käyttää Yvel:iä niin anna sille aina aluksi näin funktio

#Esimerkki arvot ja virheet.
U_arvot = np.array([1,2,3,4,5,6]) 
I_arvot = np.array([6,5,4,3,2,1])
U_err = U_arvot*0.001
I_err = I_arvot*0.05

#Muutetaan np.column_stack() metodilla arvot ja virheet muotoon, jota numeerinen laskuri odottaa.
arvot = np.column_stack([U_arvot, I_arvot]) 
virheet = np.column_stack([U_err, I_err])

# numeric() metodi odottaa parametreikseen m x k matriisin, jossa m on mittaussarjan pituus ja k muuttujien määrä.
sigma = calc.numeric(arvot, virheet) #Tällä tavalla kutsutaan numeric() metodia.
``` 

>Huom! Kaikilla muuttujilla ja virheillä tulee olla sama muoto. Eli mittaussarjoilla sama pituus.

>Huom! Jos jollakin muuttujalla ei ole virhettä, niin käytä sen virhesarjana 0-sarjaa.

Metodi toimii vain **toisistaan riippumattomille** muuttujille.
Kattavampi dokumentaatio ja läpikäyminen löytyy englanninkielisestä veriosta.
