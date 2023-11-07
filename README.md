# NLP_sets-for-Dezero
dezeroに流行り(いまさら)の自然言語処理のモデルやアーキテクチャを追加できるように作ってます。<br>
[元論文](https://arxiv.org/abs/1706.03762)<br>
[参考文献](https://qiita.com/halhorn/items/c91497522be27bde17ce)<br>
開発状況:<br>
✓GELU<br>
✓Embedding<br>
✓4次元,2次元テンソル用Matmul<br>
✓Multi Head Attention<br>
✓Self Attention<br>
✓Feed Forward Network<br>
✓Simple Attention(消去済み)<br>
✓Positional Encoding<br>
✓Layer Normalization<br>
✓Transformer Encoder<br>
✓Transformer Decoder<br>
✓Transformer Model<br>
✘BERT<br>
△GPT<br>
✓Multi Head Retention(Parallel,recurrent)<br>
✓RetNet<br>
=========学習用ファイル=======<br>
✓テキスト<=>index<br>
✓padding追加&バッチサイズごとに保存<br>
✘サブワード分割<br>
========GPU使用時の注意=======<br>
使用しているdezeroというフレームワークの都合上、そのまま逆伝播を計算しようとするとスライスを使用している箇所(Embeddingなど)でエラーが起きます.<br>
なので、cupyxをインポートして使用してください.(colabでは動作確認済み)<br>
[詳細](https://www.arbk.net/wp/%e4%bb%8a%e6%97%a5%e3%82%82%e4%b8%80%e6%97%a5%e3%81%82%e3%82%8a%e3%81%8c%e3%81%a8%e3%81%86-470/)<br>
