# Data-mining sleep brain signals using machine-learning: Exploring the effect of caffeine on EEG network dynamics
Analysis of the effect of caffeine on quantitative sleep EEG utilizing statistics and machine learning in a data-driven manner.

## Installation
Clone this repository and install the dependencies from `requirements.txt`:
```bash
git clone git@github.com:PhilippThoelke/caffeine-sleep.git
cd caffeine-sleep
pip install -r requirements.txt
```
We used a modified version of MNE-Python (0.19) for some of the visualizations, which is automatically installed when using our `requirements.txt` file. The modifications made to the original code are available [here](https://github.com/PhilippThoelke/mne-python.git).

## Usage
### Preprocessing
Scripts for running the preprocessing pipeline are located in the `preprocessing` directory.
1. Run`ExtractFeatures.py` to extract features from the raw EEG. Before running, adjust the global variables at the top of the script accordingly. The script is able to load the data in two different formats, based on the `SPLIT_STAGES` variable: when set to true, the script expects raw EEG and corresponding hypnograms as `.npy` files. If set to false, data that was previously split into sleep stages will be loaded (`ExtractRawSamples.py` can be used to split the data into sleep stages without extracting features). The data is also expected to be in `.npy` format with the following naming scheme: `<subject-id>_<sleep-stage>_*.npy`.
2. Compute differences in sample count between the awake (AWA) and wake after sleep onset (WASO/AWSL) using `ComputeSampleDifferences.py`. The script will save a file called `sample_difference<caffeine-dose>.pickle`, which is required for the next step.
3. Run `CombineFeatures.py` to group the extracted features from all subjects into a single file and perform normalization, as well as average across subjects. The resulting files containing averaged features, condition labels and subject labels are called `data_avg.pickle`, `labels_avg.pickle` and `groups_avg.pickle` respectively. These files will be used for analysis.

### Analysis
The analysis is split up into three parts: statistics, single-feature machine learning and multi-feature machine learning. The corresponding files can be found in the `statistics`, `singleFeatureML` and `multiFeatureML` directories.
1. Statistics:\
The `Statistics.ipynb` notebook contains the code used for the statistical analysis of the caffeine vs. placebo condition for all features. It runs permutation t-tests and subsequently generates a figure, showing the statistical results visually.

2. single-feature ML:
#### TODO

3. multi-feature ML:
#### TODO

## Related work
- [Caffeine Caused a Widespread Increase of Resting Brain Entropy](https://www.nature.com/articles/s41598-018-21008-6)

- [Caffeine-induced global reductions in resting-state BOLD connectivity reflect widespread decreases in MEG connectivity](https://www.frontiersin.org/articles/10.3389/fnhum.2013.00063/full)

- [Challenging sleep in aging: the effects of 200mg of caffeine during the evening in young and middle‐aged moderate caffeine consumers](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2869.2006.00518.x)

- [Effects of amphetamine, diazepam and caffeine on polysomnography (EEG, EMG, EOG)-derived variables measured using telemetry in Cynomolgus monkeys](https://reader.elsevier.com/reader/sd/pii/S1056871914002159?token=84C565DED7C251D79BAC82A61144C97174EA8B815C963E8E71A6BA54FC9A05544384B0932C822E41EDC09FF44C0A7419)

- [Effects of caffeine on daytime recovery sleep: A double challenge to the sleep–wake cycle in aging](https://www.sciencedirect.com/science/article/pii/S1389945709000094)

- [Sleep is more sensitive to high doses of caffeine in the middle years of life](https://journals.sagepub.com/doi/full/10.1177/0269881115575535?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed)
