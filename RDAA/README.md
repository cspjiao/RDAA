# RDAA

## 实验运行：
- 模型的输入依赖与[RolX](https://github.com/benedekrozemberczki/RolX)方法的特征提取，
- Rolx数据处理：
    - usa-flights:
    ```bash
    python2.7 src/main.py --input ../dataset/usa-flights.edge --embedding-output ../cache/embeddings/usa-flights_embedding.csv --recursive-features-output ../cache/features/usa-flights_features.csv --dimensions 128 --bins 4 --recursive-iterations 3 --pruning-cutoff 0.95
    ```

- 模型训练：
    - default
    ```bash
    python role.py train
    ```