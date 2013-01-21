function hog = readHogTxt(filename,rows,cols)
%readHogTxt: reading Hog txt files as written by the C++ code

%NOTE: This function is marked for change in TODO.
hog2D = dlmread(filename,' ');
hog = reshape(hog2D,rows,cols,31);
end