# Procedencia de este código

`model.py` y `resnet.py` en esta carpeta son una copia, con una única
adaptación de import (`from resnet import Resnet18` → `from .resnet import
Resnet18` para que funcione como paquete), del repositorio:

- https://github.com/zllrunning/face-parsing.PyTorch

Licencia: MIT (ver `LICENSE` en esta misma carpeta — se mantiene intacta
como exige la licencia). Esa licencia permite uso comercial, a diferencia
de otras alternativas evaluadas (ver CLAUDE.md, sección "Segmentación de
pelo").

Arquitectura: BiSeNet (Bilateral Segmentation Network) con backbone
ResNet-18, entrenado sobre CelebAMask-HQ para segmentación facial en 19
clases. El índice de clase 17 corresponde a "hair" (pelo) — ver el mapeo
completo en `hair_segmentation.py`.

Los pesos pre-entrenados NO se incluyen en este repo (son varios cientos
de MB y están alojados en Google Drive por el autor original). Ver
`download_weights.py` en la carpeta padre para descargarlos.
