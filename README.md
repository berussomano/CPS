**Script I developed for my master’s thesis. August 2025**

The script uses data with a 3-hour temporal resolution from CFSR (Climate Forecast System Reanalysis) and the cyclone tracking database by Gramcianinov et al. (2020). The respective access links are included in the script.

The script selects, classifies, and plots cyclone composites in polar coordinates!  
Classification is based on the methodology of Hart (2003), the Cyclone Phase Space (CPS), and the definition of bombs by Sanders & Gyakum (1980).
The CPS code for case studies and educational purposes made availabre by Frederic Ferry was very helpfull in the begining and I aknowlegde him for that. I recommend a check on this one: https://github.com/fredericferry/era5_cps_diagram 

Attention to the shape and format of the CFSR data. It is showed in the script.
For the Gramcianinov et al. (2020) cyclone tracking database, just use the function rm.le_tracking_camargo. See the Result_Of_The_Function_le_tracking_camargo to see the format of the cyclone tracking dataframe.

For questions, suggestions, or discussions, feel free to contact me via email: berussomano@gmail.com





