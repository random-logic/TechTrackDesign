# Task 1 - Model Assessment

To assess the 2 given models, we first applied NMS with confidence threshold 0.5 and IOU threshold 0.4. When matching ground truths to detections, we used an IOU threshold of 0.5.

| Model | Precision  | Recall     | mAP        |
|-------|------------|------------|------------|
| 1     | 0.9212     | 0.3025     | 0.2561     |
| 2     | **0.9441** | **0.4135** | **0.3079** |

| Class            | Model 1 mAP | Model 2 mAP |
|------------------|-------------|-------------|
| barcode          | 0.1605      | 0.1780      |
| car              | 0.3042      | 0.2999      |
| cardboard box    | 0.2786      | 0.3629      |
| fire             | 0.0000      | 0.0000      |
| forklift         | 0.1303      | 0.2241      |
| freight container| 0.0215      | 0.0320      |
| gloves           | 0.1552      | 0.3128      |
| helmet           | 0.0745      | 0.0973      |
| ladder           | 0.0202      | 0.0557      |
| license plate    | 0.1198      | 0.1453      |
| person           | 0.0922      | 0.1328      |
| qr code          | 0.4055      | 0.4097      |
| road sign        | 0.0349      | 0.0435      |
| safety vest      | 0.0553      | 0.0792      |
| smoke            | 0.0447      | 0.0469      |
| traffic cone     | 0.1917      | 0.1972      |
| traffic light    | 0.1880      | 0.2329      |
| truck            | 0.2495      | 0.3380      |
| van              | 0.4151      | 0.4879      |
| wood pallet      | 0.0207      | 0.0261      |

Model 2 outperforms Model 1 overall, achieving higher precision (0.9441 vs. 0.9212), recall (0.4135 vs. 0.3025), and mAP (0.3079 vs. 0.2561). At the per-class level, Model 2 shows clear improvements for cardboard box, forklift, gloves, ladder, person, safety vest, traffic light, truck, and van, often with substantial gains (e.g., gloves: 0.3128 vs 0.1552, truck: 0.3380 vs 0.2495, van: 0.4879 vs 0.4151). Model 1 performs slightly better on a few classes such as car (0.3042 vs. 0.2999), but the differences are small. Both models struggle equally on classes like fire (0.0000) and freight container (mAP < 0.05). Overall, Model 2 demonstrates broader and more consistent improvements across classes, making it the stronger choice.


# Task 2 - Sampling Strategy

To ensure that the TechTrack dataset is both representative and useful for evaluating model performance, we applied a stratified balanced sampling strategy with a total of 15,000 annotations from over 5000 images.

### Criteria for Selecting Representative Subsets of Data

- **Stratified Sampling Across All Classes**
  - Sampling was stratified so that every class in the dataset was represented equally.
  - This prevents rare classes (e.g., fire, freight container, ladder) from being omitted during training or evaluation.

- **Balanced Sampling for Underperforming Classes**
  - Using per-class mAP from Model 2, we identified underperforming classes (mAP < 0.15).
  - These classes were allocated double the sampling quota compared to their natural occurrence, ensuring that the model sees more examples of the difficult categories such as freight container, smoke, and road sign.

### Justification for Why This Sampling Strategy Is Valid

- Improved Per-Class Precision: Stratification ensures that performance metrics like precision and recall are meaningful on a per-class basis, since each class has a sufficient number of examples.
- Bias Mitigation: Without balancing, frequent classes (e.g., wood pallet, person) would dominate the dataset, leading to inflated scores for those categories and poor generalization for rare ones. Balanced sampling corrects this by boosting underrepresented or underperforming classes.
- Alignment with Evaluation Goals: As per-class precision comparison was emphasized in task 1, this strategy directly supports fairer evaluation across categories.

### Sanity Check

| Class            | Sampled Count | Sampled % | Unsampled Count | Unsampled % | Difference (Sampled % - Unsampled %) |
|------------------|---------------|-----------|-----------------|-------------|-------------------------------------|
| barcode          | 73            | 0.23      | 283             | 0.77        | -0.54                               |
| car              | 714           | 2.29      | 1379            | 3.76        | -1.47                               |
| cardboard box    | 4842          | 15.53     | 4995            | 13.60       | 1.93                                |
| fire             | 2422          | 7.77      | 2793            | 7.61        | 0.16                                |
| forklift         | 622           | 1.99      | 1103            | 3.00        | -1.01                               |
| freight container| 245           | 0.79      | 318             | 0.87        | -0.08                               |
| gloves           | 131           | 0.42      | 256             | 0.70        | -0.28                               |
| helmet           | 2122          | 6.81      | 2170            | 5.91        | 0.90                                |
| ladder           | 229           | 0.73      | 277             | 0.75        | -0.02                               |
| license plate    | 264           | 0.85      | 359             | 0.98        | -0.13                               |
| person           | 5920          | 18.99     | 6368            | 17.34       | 1.65                                |
| qr code          | 133           | 0.43      | 369             | 1.00        | -0.57                               |
| road sign        | 609           | 1.95      | 720             | 1.96        | -0.01                               |
| safety vest      | 1226          | 3.93      | 1260            | 3.43        | 0.50                                |
| smoke            | 1137          | 3.65      | 1495            | 4.07        | -0.42                               |
| traffic cone     | 287           | 0.92      | 506             | 1.38        | -0.46                               |
| traffic light    | 497           | 1.59      | 1193            | 3.25        | -1.66                               |
| truck            | 246           | 0.79      | 782             | 2.13        | -1.34                               |
| van              | 281           | 0.90      | 765             | 2.08        | -1.18                               |
| wood pallet      | 9179          | 29.44     | 9330            | 25.41       | 4.03                                |

The distribution in the sampled classes remains approximately the same as the original. Becasue we first sampled annotations and then selected the corresponding images (which may contain additional annotations outside the sampled set), the resulting image-level class percentages differ slightly from what stratified balanced sampling would ideally produce. Most classes differ by less than 2%, with the main exception being wood pallet, which is oversampled by about 4%. This is acceptable since its per-class mAP was relatively low, and the extra representation can help improve performance. Overall, the approach ensures rare and underperforming classes are more visible while maintaining the minimum image count requirement.


# Task 3 - Choosing NMS IOU Threshold

To evaluate the effect of NMS IOU thresholds, we compared mAP across thresholds from 0.4 to 0.9 using Model 2 (the best performing model from Task 1).

| Threshold | mAP        |
|-----------|------------|
| 0.4       | 0.2575     |
| 0.5       | 0.2588     |
| 0.6       | **0.2595** |
| 0.7       | 0.2569     |
| 0.8       | 0.2473     |
| 0.9       | 0.2224     |

The results show that performance peaks at a threshold of 0.6, achieving the highest mAP (0.2595). Thresholds below 0.6 slightly underperform, while thresholds above 0.6 lead to a clear decline in mAP, indicating that higher thresholds become too restrictive and discard true positives. Therefore, setting the NMS threshold to 0.6 provides the best balance between suppressing duplicate detections and retaining correct ones.


# Task 4 - Data Augmentation

The following table reports overall mAP and per-class mAPs for three augmentations: Gaussian Blur, Vertical Flip, and Adjust Brightness.

| Class            | Gaussian Blur | Vertical Flip | Adjust Brightness |
|------------------|---------------|---------------|-------------------|
| **Overall mAP**  | 0.1886        | 0.0373        | 0.2507            |
| barcode          | 0.1779        | 0.0841        | 0.1484            |
| car              | 0.1390        | 0.0059        | 0.1753            |
| cardboard box    | 0.3387        | 0.0201        | 0.3624            |
| fire             | 0.0000        | 0.0000        | 0.0000            |
| forklift         | 0.1665        | 0.0000        | 0.1996            |
| freight container| 0.0200        | 0.0119        | 0.0282            |
| gloves           | 0.1716        | 0.0010        | 0.1942            |
| helmet           | 0.0263        | 0.0011        | 0.0928            |
| ladder           | 0.0324        | 0.0050        | 0.0454            |
| license plate    | 0.0937        | 0.0015        | 0.1149            |
| person           | 0.0662        | 0.0008        | 0.1307            |
| qr code          | 0.3310        | 0.0320        | 0.3780            |
| road sign        | 0.0261        | 0.0054        | 0.0411            |
| safety vest      | 0.0422        | 0.0016        | 0.0710            |
| smoke            | 0.0297        | 0.0000        | 0.0379            |
| traffic cone     | 0.0944        | 0.0015        | 0.1308            |
| traffic light    | 0.1744        | 0.0280        | 0.1884            |
| truck            | 0.1545        | 0.1058        | 0.2554            |
| van              | 0.2260        | 0.0303        | 0.3697            |
| wood pallet      | 0.0170        | 0.0080        | 0.0214            |


This table reveals clear differences in how augmentations affect model performance:

- **Gaussian Blur** reduces overall mAP from the baseline (0.3079 in Task 1) to 0.1886. The drop indicates that the model struggles with blurred inputs, particularly for small or detail-dependent objects such as license plates (0.0937 vs 0.1453 in Task 1) and helmets (0.0263 vs 0.0973 in Task 1), which rely on fine-grained features. However, relatively structured classes like cardboard box (0.3387 vs 0.3629 in Task 1) and QR code (0.3310 vs 0.4097 in Task 1) remain more resilient.

- **Vertical Flip** severely degrades performance (overall mAP = 0.0373). This is expected since many objects in the dataset (cars, license plates, people) have strong orientation, and vertical flipping creates unrealistic examples that confuse the detector. Only a few classes such as truck (0.1058 vs 0.3380 in Task 1) retain marginal detection ability, while most others collapse close to zero mAP.

- **Adjust Brightness** achieves the best robustness among the three augmentations, with an overall mAP of 0.2507. While still lower than the baseline of 0.3079, it preserves performance reasonably well across most classes. Some examples include cardboard box (0.3624 vs 0.3629 in Task 1) and QR code (0.3780 vs 0.4097 in Task 1). This indicates the model can handle brightness variations reasonably well, but its accuracy is not fully preserved.

**Conclusion:** The model is moderately robust to brightness variation, vulnerable to blur, and highly sensitive to unnatural orientation changes such as vertical flips. These findings imply that including realistic augmentations (e.g., brightness adjustments) could improve generalization, while unrealistic ones (vertical flips) should be avoided.


# Task 5 - Hard Negative Mining

We analyzed how different lambda (λ) values in Hard Negative Mining (HNM) influence the selection of images. Our procedure selects the top 1000 images with the highest loss per configuration. We tested four lambda configurations:

- **Control**: λ = (0.33, 0.33, 0.33, 1)
- **Emphasize Location**: λ = (1, 0.33, 0.33, 1)
- **Emphasize Objectness**: λ = (0.33, 1, 0.33, 1)
- **Emphasize Class**: λ = (0.33, 0.33, 1, 1)

By varying these parameters, we observe how the weighting of bounding box regression, objectness, and class losses affects which types of annotations contribute the hardest negatives, directly affecting which images get selected. For example, emphasizing location prioritizes images where bounding box errors are largest, while emphasizing class increases the selection of images with misclassified objects. This analysis helps identify which classes are most affected by what type of loss.

### Analysis

The following table shows the distribution of classes within the top 1000 hardest negative images for each lambda configuration. Absolute counts indicate how many images contained a specific class.

| Class            | Control Count | Location Count | Objectness Count | Class Count |
|------------------|---------------|----------------|------------------|-------------|
| barcode          | 3             | 3              | 3                | 3           |
| car              | 77            | 78             | 75               | 81          |
| cardboard box    | 181           | 183            | 182              | 178         |
| fire             | 13            | 14             | 12               | 11          |
| forklift         | 75            | 84             | 73               | 69          |
| freight container| 15            | 16             | 15               | 16          |
| gloves           | 17            | 16             | 17               | 18          |
| helmet           | 323           | 317            | 321              | 329         |
| ladder           | 17            | 17             | 19               | 19          |
| license plate    | 25            | 24             | 26               | 27          |
| person           | 512           | 505            | 511              | 517         |
| qr code          | 8             | 8              | 8                | 8           |
| road sign        | 30            | 31             | 30               | 31          |
| safety vest      | 268           | 266            | 266              | 274         |
| smoke            | 14            | 15             | 14               | 11          |
| traffic cone     | 18            | 16             | 19               | 18          |
| traffic light    | 27            | 20             | 27               | 28          |
| truck            | 21            | 26             | 20               | 21          |
| van              | 22            | 26             | 21               | 22          |
| wood pallet      | 166           | 159            | 168              | 164         |

From the table, we can observe how varying lambda values shifts the emphasis of HNM sampling and impacts specific classes:

- **Emphasizing Location** increases the representation of classes where bounding box errors are largest. For example, images containing the forklift class (75 → 84) are more frequently selected compared to the control configuration. This is likely due to its large size and complex shape, which can lead to higher localization errors. In contrast, other classes, like wood pallet (166 → 159), are smaller and easier to detect, so fewer images containing this class are selected. Similarly, images with helmets (323 → 317) are often small and well-centered, resulting in fewer selections for this class.

- **Emphasizing Objectness** shifts focus toward classes with uncertain objectness predictions. The counts are similar to the control configuration, with classes usually deviating by at most 2 observations. This indicates that all classes have comparable objectness detection losses. As no class is disproportionately affected, image selection is mostly unchanged.

- **Emphasizing Class** increases the selection of images prone to misclassification. Notably, images containing helmets (323 → 329) and safety vests (268 → 274) have higher representation, indicating that these examples are harder to classify. Other classes may not be significantly easier to classify, as no class experienced a substantial decrease in representation compared to the control configuration; the largest decrease was only 3 images for a single class.

Adjusting λ values in HNM enables targeted sampling, helping the model focus on localization, objectness, or classification for specific classes and images.
