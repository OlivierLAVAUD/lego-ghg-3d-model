# Notes techniques : format LEGO GHG / NU20

Ce document resume le format binaire des fichiers `.ghg` PC etudies dans ce depot, ainsi que
l'architecture du parseur. Les documents `jaandev_*.md` decrivent une variante big-endian
(GameCube/Wii) et ne s'appliquent pas directement aux fichiers PC de ce depot.

## 1. Conteneur

- Signature : `NU20`.
- Endianness : little-endian.
- Chunks : `[tag 4CC][u32 size][data]`.
- Le champ `size` **inclut** l'en-tete de 8 octets : `next = offset + size`.
- Le champ taille de `NU20` n'est pas un chunk standard (valeur negative/particuliere). La chaine
  principale commence a `0x10`.
- `DISP` est imbrique dans `CDAT`. La profondeur des chunks est calculee en comparant les plages
  `[data_offset, offset + size)`.

Tags connus : `HEAD`, `NTBL`, `TREF`, `TST0`, `MS00`, `INID`, `FDNS`, `BNDS`, `CDAT`, `VBIB`,
`DISP`, `SALI`, `DYNO`, `GSNH`, `PNTR`.

Le parseur lit d'abord la chaine lineaire, puis complete avec les tags connus trouves isoles
(validation de taille et de bornes), car certains chunks ne sont pas dans la chaine principale.

## 2. Textures DDS

- Les textures sont embarquees apres les metadonnees.
- Chaque texture commence par `DDS ` et un en-tete de 128 octets.
- Formats observes : DXT1 (8 octets / bloc 4x4) et DXT5 (16 octets / bloc 4x4).
- La taille totale est calculee mipmap par mipmap. Le parseur avance ensuite de cette taille pour
  eviter de detecter de faux `DDS ` dans les donnees compressees.
- Les textures sont converties en PNG via Pillow et exportees dans un ZIP.

## 2bis. Textures et materiaux

- `TST0` contient une table de descripteurs de `0x80` octets :
  - `+0x00` largeur, `+0x04` hauteur ;
  - `+0x38` format (`1` = DXT1, `6` = DXT5) ;
  - `+0x40` nombre de mipmaps, `+0x44` taille des donnees.
- `MS00` commence par `u32 materialCount` ; les records font `0x2C4` octets et commencent a
  l'offset `8` :
  - `+0x38` identifiant du materiau ;
  - `+0x54` couleur diffuse RGBA (4 floats) ;
  - `+0x74` index de texture signe (`-1` = aucune texture), suivi d'un second index.
- `GSNH` associe les layers aux os et aux parties. Apres le nombre de parties, un pointeur mene
  a la table des index de materiaux ; cette table est **inversee** pour retrouver l'ordre des
  draws.
- Le chunk `TREF` est vide dans les fichiers PC etudies ; le parseur l'ignore dans ce cas.
- L'export GLB cree un `PBRMaterial` par materiau avec `baseColorTexture`, la couleur diffuse
  MS00 et `alphaMode = BLEND` lorsque la texture contient de la transparence.
- trimesh inverse l'axe V lors de l'export glTF : les UV sont pre-flippes avant
  `TextureVisuals` pour conserver l'orientation des textures DDS dans le GLB.

## 3. Section geometrie

La geometrie est stockee **en fin de fichier**, apres la derniere texture DDS et les chunks
`PNTR` / `GSNH` / `DYNO` / `SALI` / `VBIB` :

```
[u16 nVB]
[(u32 byteSize, data) * nVB]
[u16 nIB]          # observe : 1
[u32 ibBytes]
[u16 indices...]   # ibBytes / 2
```

La section se termine exactement a EOF. Le parseur essaie les offsets candidats
`after_textures + 0/2/4/6` puis, en dernier recours, scanne le fichier et garde la section dont la
somme des tailles de vertex buffers est maximale.

## 4. Vertex buffers

Chaque vertex buffer est une suite de sommets de stride variable. Layout observe :

| Stride | Position | Normale | Tangente | UV |
|-------:|---------:|--------:|---------:|----:|
| 28 | 0 | 12 | 16 | 20 |
| 32 | 0 | 12 | 16 | 24 |
| 36 | 0 | 12 | 16 | 20 |
| 40 | 0 | 12 | 16 | 24 |
| 44 | 0 | 12 | 16 | 28 |

- Position : `float32[3]`.
- Normale : 4 octets signes, decodes par `b / 127.5 - 1.0` ; les 3 premiers octets sont utilises.
- UV : `float32[2]`.
- Le stride 36 ajoute 8 octets apres les UV (couleurs de sommets), sans deplacer les UV.

Les offsets UV ont ete verifies sur les echantillons ; ceux du stride 36 sont en `20` (et non
`28`, qui contient les couleurs).

## 5. Enregistrements DISP

Les sous-maillages sont decrits par des enregistrements consecutifs de **56 octets** (`0x38`).

| Offset | Champ |
|-------:|-------|
| `+0x00` | attributs (observe : 6) |
| `+0x04` | nombre d'indices |
| `+0x08` | stride (u16) |
| `+0x0C` | flags / attributs |
| `+0x10` | flags / attributs |
| `+0x14` | offset dans le vertex buffer (en sommets) |
| `+0x18` | nombre de sommets |
| `+0x1C` | offset dans l'index buffer |
| `+0x24` | index du vertex buffer (0-based) |

Entre deux draws consecutifs, l'ecart
`ibOffset[k+1] - (ibOffset[k] + indexCount[k])` vaut generalement `0`, `1`, `2` ou `8`.
Le parseur accepte un gap de `0..8` pour reconstruire les runs.

## 6. Reconstruction des faces

- Les indices de chaque draw sont **relatifs au `vbOffset`** du draw.
- Les faces sont des **triangle strips**. Pour chaque triplet d'indices consecutifs :
  - pas pair : `[a, b, c]`
  - pas impair : `[a, c, b]`
- Les triangles degeneres (`a == b`, `b == c`, `a == c`) sont supprimes.
- Les indices doivent rester dans `[0, vbCount)`.
- Le winding est verifie par le produit scalaire entre la normale geometrique et la normale
  stockee. Les valeurs observees sont :

| Modele | Draws | Sommets | Faces | Dot normales |
|--------|------:|--------:|------:|-------------:|
| scooter_pc | 8 | 920 | 1418 | 0.9297 .. 1.0 |
| biplane_pc | 44 | 4757 | 7183 | 0.9861 .. 1.0 |
| indianajones_dj_pc | 19 | 4605 | 8752 | 0.9008 .. 0.996 |

`trimesh.Trimesh(process=True)` recalcule les normales pour l'export GLB.

## 6bis. Transforms GSNH

Le chunk `GSNH` decrit un petit squelette et une table de parties :

- en-tete : `numberImages`, pointeur vers les metadonnees d'images, pointeur vers `MESH`,
  `numberBones`, offsets des trois tables d'os, `numberLayer`, offset des layers ;
- chaque pointeur est un **offset signe relatif au champ qui le contient** ;
- table 1 : `numberBones` records de `0x60` octets (matrice 4x4 locale, pointeur de nom,
  index du parent) ;
- table 2 : `numberBones` matrices 4x4 locales ; elles se composent avec le parent
  (`world = local @ parentWorld`, vecteurs lignes) pour obtenir la pose de bind ;
- table 3 : matrices inverses de bind (non utilisees pour l'export) ;
- layers : jusqu'a quatre tables (`p1`..`p4`) ; chaque entree associe un os a un nombre de
  parties. L'ordre des parties suit celui des records `DISP`.
- chaque partie possede un index de materiau dans une table dediee ; elle est inversee pour
  correspondre a l'ordre des draws.

Les sommets de chaque partie sont exprimes dans l'espace local de l'os : le parseur applique
`world` aux positions et la rotation 3x3 aux normales avant la deduplication. Les parties
sans os (`bone = -1`) sont deja en espace monde.

Indiana Jones possede 3 records isoles avant la suite principale ; leur nombre total (19)
correspond aux 19 parties GSNH, ils sont donc conserves. Pour le scooter, un record
parasite est ignore car le total ne correspond pas aux 8 parties GSNH.

## 7. Cas particuliers des echantillons

### scooter_pc.ghg

- 1 vertex buffer, stride 28, `@0x651A`, 1910 sommets.
- Index buffer `@0x13608`, 2642 indices.
- 8 draws.
- 2 textures.

### biplane_pc.ghg

- 2 vertex buffers :
  - block 0 : stride 28, `@0x3707A`, 15184 sommets ;
  - block 1 : stride 32, `@0x9ED3E`, 2168 sommets.
- Le vertex buffer 0 contient **deux copies identiques** du maillage :
  - sommets `0..7592` et `7592..15184` ;
  - ecart maximal des positions : `3.7e-9`.
- Les faces dupliquees sont supprimees en arrondissant les positions a 4 decimales.
- 44 draws apres filtrage, 7183 faces.
- 24 textures.

### indianajones_dj_pc.ghg

- 3 vertex buffers :
  - block 0 : stride 36, `@0x17C78A`, 655 sommets (3 parties isolees, deja en espace monde) ;
  - block 1 : stride 28, `@0x1823AA`, 4868 sommets ;
  - block 2 : stride 36, `@0x1A381E`, 2547 sommets.
- Les records stride 28 utilisent `vbIndex=1`, les records stride 36 utilisent `vbIndex=2`.
- Indices `0..1380` et 2 indices finaux inutilises.
- 19 draws (3 parties isolees + 16 draws principaux), 4605 sommets, 8752 faces.
- Le personnage est deja en orientation Y-up dans le repere du fichier ; aucun flip automatique
  n'est applique.
- Les os `Head`, `LeftArm`, `RightArm`, `LeftUpLeg`, `RightUpLeg`, `Spine` et `Hips`
  deplacent leurs parties locales vers la pose de bind.
- 4 textures.

## 8. Architecture

- `GHGModel(filepath_or_bytes)` :
  - `_read_chunks()` : chaine + tags isoles + profondeur ;
  - `_extract_ntbl()` : chaine NTBL, avec saut du compteur u32 initial ;
  - `_extract_textures()` : DDS -> PNG ;
  - `_parse_tst0()` : descripteurs de textures ;
  - `_parse_ms00()` : materiaux, couleurs diffuses et index de textures ;
  - `_extract_geometry()` : section EOF, vertex buffers, records DISP, strips, dedup,
    materiaux par face ;
  - `_parse_gsnh_parts()` : os, matrices de bind, mapping os -> parties -> materiaux ;
  - `_read_draw_records()` : detection du run DISP et des records isoles ;
  - `_build_textured_scene()` : un `trimesh.Trimesh` par materiau avec UVs et texture ;
  - `_build_stats()` : statistiques pour l'UI ;
  - `export_glb()`, `export_obj()`, `export_textures_zip()`.
- `app.py` : interface Gradio, tableaux chunks/draws, galerie de textures, exports.
- `tests/test_ghg_parser.py` : tests de non-regression.

## 9. Limites et pistes

- Les transforms de bones sont appliquees pour la pose de bind ; les animations ne sont pas lues.
- Les variantes console big-endian ne sont pas supportees.
- Le seul mapping VB/IB connu est celui des echantillons ; les fichiers avec plusieurs index
  buffers ou des records non consecutifs pourraient necessiter des ajustements.
