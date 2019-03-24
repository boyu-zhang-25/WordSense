# WordNet WSD v1.0

This archive contains data from the experiments described in Section 6 of the following paper.

White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. [Universal decompositional semantics on universal dependencies](http://aswhite.net/media/papers/white_universal_2016.pdf). To appear in *Proceedings of the Conference on Empirical Methods in Natural Language Processing 2016*.

If you make use of this data set in a presentation or publication, we ask that you please cite this paper.

The column descriptions and values for `wsd_eng_ud1.2_10262016.tsv` can be found below. 

| Column            | Description       | Values            |
|-------------------|-------------------|-------------------|
| Split             | The train-dev-test split from the English Web Treebank | `train`, `dev`, `test` |
| Annotator.ID      | The annotator that provided the response | `0`, ..., `1053` |
| Sentence.ID       | The file and sentence number of the sentence in the English Universal Dependencies v1.2 treebank with the format `LANGUAGE-CORPUS-SPLIT.ANNOTATION SENTNUM` |  |
| Arg.Token         | The position of the argument in the sentence starting at zero |  |
| Arg.Lemma         | The argument lemma        |  |
| Synset            | WordNet Synset ID         |  |
| Sense.Definition  | WordNet Synset Definition |  |
| Sense.Response    | Whether the word sense is chosen by the annotator |`True`, `False`|
| First.Synset      | The first Synset in WordNet for the argument|  |
| Display.Position  | The order of options (synsets) that were displayed to annotators|  |
