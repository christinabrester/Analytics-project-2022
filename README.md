# Analytics-project-2022
<b>Load control detection using power quality data</b>

The code was used to generate the experimental results presented at the 27th International Conference on Electricity Distribution, CIRED 2023: 
<br>
<i>Automated Load Control Detection Using Power Quality Data And Machine Learning</i>
<br>by Christina Brester(1), Antti Hildén(2), Mikko Kolehmainen(1), Pertti Pakonen(2), Harri Niska(1)
<br>(1) – University of Eastern Finland – Finland
<br>(2) – Tampere University – Finland

<br>
<br>
In the article, we have introduced two scenarios for load control detection: supervised and unsupervised. In each scenario, we have tested four different sets of inputs: first three include only measurement data (power and power quality), whereas the last one has been generated using the MLP-based feature engineering. The source code consists of the following parts: 
<br>1 - <b>MLP_nowcasting.py</b> used to train a number of MLPs with different architectures that nowcast power quality variables;
<br>2 - <b>parallel_run.py</b> used to run MLP_nowcasting.py in parallel;
<br>3 - <b>feature engineering (use the best MLP).ipynb</b> used to post-process MLP-training: select the best model based on the validation performance, then, using the selected architectures, generate new features - estimates and deviations;
<br>4 - <b>example of nowcasting.html</b> illustrates how MLP estimated power quality: https://christinabrester.github.io/Analytics-project-2022/example_of_nowcasting
<br>5 - <b>control detection based on measurement data.ipynb</b> used to test the supervised and unsupervised scenarios on the measurement data;
<br>6 - <b>control detection with feature engineering.ipynb</b> used to test the supervised and unsupervised scenarios after the feature engineering step.
