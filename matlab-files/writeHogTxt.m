function writeHogTxt(hog, outputFilename)
%writeHogTxt: writing HoG feature(hog) to Txt file(outputFilename)
%Given an input 3D hog (mxnx31), this function writes it to a space
%delimited ASCII file. This file can be read by the C++ function
%readHogFromFile

%NOTE: This function is marked for change in TODO.
dlmwrite(outputFilename,hog,'delimiter',' ','precision','%0.6f');

end

