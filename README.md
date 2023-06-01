# AIHazardsFinder
A screening tool for unknown chemical hazards in food.
Operating instructions of AIHazardsFinder software
The AIHazardsFinder software is easy to operate without programming foundation. After selecting and loading mass spectrometry file and peak table file, click the "AI Classification" button to perform the classification task, and results will be automatically exported and saved in the folder. In addition, we have added molecular formula calculation step in the software, which can automatically perform molecular formula calculation.
1)	Double-click the AIHazardsFinder.exe to open the software.
2)	File loading. We select a file through pull-down menu of the “File” button in the menu bar, including mass spectrometry file in mzXML or MGF format, and peak table file in excel format (Note: mass spectrometry file in mzXML or mgf format files, peak table files do not have to be provided).
3)	Click the "AI Classification" button, and the software starts to classify unknown MS/MS spectra. When progress bar reaches 100%, the classification task is completed, and classification results are exported to the default folder.
4)	Molecular formula calculation and filtration. Click the "Formula Filtration" button, the software starts to perform molecular formula calculation and filtration. The software starts to calculate formulas of suspect compounds, which were screened out in step 3. If formula cannot be calculated, the compound is not retained.
In addition, we can also switch the extracted ion chromatograms, MS/MS spectra displayed in the visualization window on the right by clicking “Last”, “Next”, or right-clicking the left table and selecting the “Detailed information” prompt box.

The trained model cannot be uploaded to Github due to its large size. Please feel free to contact me via email. You can also train your own model through the provided code. Email: chentiantian@dicp.ac.cn. I will definitely reply to you within 24 hours.
