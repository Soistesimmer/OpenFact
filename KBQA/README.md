# KBQA + OpenFact
This directory contains code for KBQA results. We generally follow [UniK-QA](https://github.com/facebookresearch/UniK-QA) on WebQSP but enrich the model input with OpenFact KN. 

To get this part of model inputs. First, get the entity wikidata id with their freebase id using wikipedia function or an [avalible DB](http://storage.googleapis.com/freebase-public/_fb2w_._nt_.gz).
> 1_search_entity.py
> 2_fid2wid.py

Next, get all entity pairs appeared in questions of WebQSP, then retrieve corresponding OpenFact KN.
> 3_prepare_ranking_data.py
> 4_prepare_openfact_kn.py

Finally, select the best-matching retrieved KN using DPR by comparing their embeddings and a question embedding.
> 5_prepare_topk_openfact_kn.py

After getting the OpenFact KN for each question through the above steps. The final model based on FiD can be trained following [UniK-QA](https://github.com/facebookresearch/UniK-QA).