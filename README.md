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
<br>1 - <i>MLP_nowcasting.py</i> used to train a number of MLPs with different architectures that nowcast power quality variables;
<br>2 - <i>parallel_run.py</i> used to run MLP_nowcasting.py in parallel;
<br>3 - <i>feature engineering (use the best MLP).ipynb</i> used to post-process MLP-training: select the best model based on the validation performance, then, using the selected architectures, generate new features - estimates and deviations;
<br>4 - <i>example of nowcasting.html</i> illustrates how MLP estimated power quality 
<br>5 - <i>control detection based on measurement data.ipynb</i> used to test the supervised and unsupervised scenarios on the measurement data;
<br>6 - <i>control detection with feature engineering.ipynb</i> used to test the supervised and unsupervised scenarios after the feature engineering step.
