inpDir='../sample-data/exemplars/exemplar-mat-files/';
outDir = '../sample-data/exemplars/exemplar-txt-files/';

outFile = '../sample-data/exemplars/exemplar-txt-files-list';
descFileDir = '../sample-data/exemplars/';
files = dir(inpDir);
numFiles = size(files,1);
fh = fopen(outFile,'w');
currDir=pwd;

useRelativePaths=true;
descRelativePathPrefix='../sample-data/exemplars/';
exRelativePathPrefix='../sample-data/exemplars/exemplar-txt-files/';

mkdir(outDir);
for i=3:numFiles
    if(files(i).isdir==1)
        continue;
    end
	fname = files(i).name;
    baseName = strrep(fname,'.mat','');	
	inpName = strcat(inpDir,fname);
	models = loadSingleVariableMAT(inpName);
	numWs = size(models,2);
    fprintf('Processing %s\n',inpName);
    
    if(useRelativePaths==false)
        fprintf(fh,'%s %s\n',baseName,fullfile(currDir,baseName));
    else
        relPath = sprintf('%s%s',descRelativePathPrefix,baseName);
        fprintf(fh,'%s %s\n',baseName,relPath);
    end
    
    descFile = sprintf('%s%s',descFileDir,baseName);
    fclass = fopen(descFile,'w');

	for j=1:numWs
	  model = models{j}.model;
	  w = model.w;
	  b = model.b;
      [m,n,d] = size(w);
      outName = sprintf('%s%s_%02d.txt',outDir,baseName,j);
      if(useRelativePaths==false)          
          outName = fullfile(currDir,outName);
      else
          outDescName = sprintf('%s%s_%02d.txt',exRelativePathPrefix,baseName,j);
      end
      fprintf(fclass,'%s %d %d %0.6f\n',outDescName,m,n,b);
      writeHogTxt(w,outName);
      
    end
    fclose(fclass);
    
end
fclose(fh);
