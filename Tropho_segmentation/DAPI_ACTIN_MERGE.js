
var ImagePlus = org.imagej.ImagePlus;
importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.process.ImageProcessor);
importClass(Packages.ij.plugin.ChannelSplitter);
importClass(Packages.ij.plugin.RGBStackMerge);

// folders based on treatment
var inputFolder = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/UCLA-DOX2/";
var outputFolder = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/UCLA-DOX2_COMPRESSED/";

// Function to list files in the folder
function getFileList(inputFolder) {
    importClass(Packages.java.io.File);
    var folder = new File(inputFolder);
    var files = folder.list(); // Returns the list of file names
    return files;
}

function splitChannels() {
  var channels = ChannelSplitter.split(imp);

  for (var i = 0; i < channels.length; i++) {
    var channel = channels[i];
    channel.show();
  }
}

function processFile(inputFolder, outputFolder, file1, file2) {
	
    // Construct the full file paths for both files
    var fullPath1 = inputFolder + file1;  
    var fullPath2 = inputFolder + file2;
   

    // Open both image files
   	var imp1 = IJ.openImage(fullPath1);  // Opens the first image
	var imp2 = IJ.openImage(fullPath2);  // Opens the second image
   	
   	if (imp1 != null) {
    var channels1 = ChannelSplitter.split(imp1);  // Split channels of first image
    for (var i = 0; i < channels1.length; i++) {
        channels1[i].show();  // Show each channel of the first image
    }
   	}


// Split the channels of the second image
if (imp2 != null) {
    var channels2 = ChannelSplitter.split(imp2);  // Split channels of second image
    for (var i = 0; i < channels2.length; i++) {
        channels2[i].show();  // Show each channel of the second image
    }
 
  

    // Extract the last number from the filenames
    var number1 = file1.match(/(\d+)$/);  // Get the last number from file1
    var number2 = file2.match(/(\d+)$/);  // Get the last number from file2
    
        // Construct the variables for the first image
        var channels1 = ChannelSplitter.split(imp1);
        var channels2 = ChannelSplitter.split(imp2);
        var merged = RGBStackMerge.mergeChannels([channels1[0], channels2[2]], false);

    // Display the merged result
    	merged.show();
        
        // Save the processed images as TIFF
    	IJ.saveAs("Tiff", outputFolder + file1); // Save image
    	return;
    }
}


function processFolder(inputFolder, outputFolder) {
	var fileList = getFileList(inputFolder);
// Process files in pairs based on the last number in their names
	for (var i = 0; i < fileList.length; i++) {
    // Get the last number from the current file
    	var currFile = (fileList[i]);
    	if (currFile.indexOf("ACTIN") !== -1) {
    		var actin_number = String(currFile).match(/ACTIN(\d+)/);
    		var actin_number = actin_number[1];
    		
            for (var j = 0; j < fileList.length; j++) {
                var potentialDAPIFile = fileList[j];
                // Check if the potential file is a DAPI file and has the same number
                if ((potentialDAPIFile.indexOf("DAPI") !== -1)) {
                	var dapi_number = potentialDAPIFile.match(/DAPI(\d+)/);
                	
                	var dapi_number = dapi_number[1];
                	if(dapi_number === actin_number){
                    // Call processFile with the found pair
                    processFile(inputFolder, outputFolder, currFile, potentialDAPIFile);
                    break;
                	}
                    }
            }
    }
    }
}

processFolder(inputFolder, outputFolder)



