// This script runs through the FIJI macro. It takes in arguments from the python input and processes the images accordingly.
// The script takes an input directory, determines if it is a parent directory or a single folder, and processes the images accordingly.
// Images are split by channel, thresholded if they are in the ACTIN channel, and merged back together to create a hyperstack of DAPI and ACTIN channels.
// The script saves the processed images as TIFF files in a new folder within the parent directory.

// Channel 1: ACTIN (dictated as channel[0])
// Channel 3: DAPI (dictated as channel[2])

// Establish environment to run the FIJI macro
var ImagePlus = org.imagej.ImagePlus;
var SPLT = Java.type("java.lang.String");
importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.process.ImageProcessor);
importClass(Packages.ij.plugin.ChannelSplitter);
importClass(Packages.ij.plugin.RGBStackMerge);
importClass(java.io.File);

// Parse the arguments provided from the python input 
var args = getArgument();
var splitArgs = new SPLT(args).split(",");
print(splitArgs);   
var parent_directory = "";
var thresh_min = 0;
var thresh_max = 0;

for (i=0; i<splitArgs.length; i++) {
    var pair = new SPLT(splitArgs[i]).split("=");
    print(pair);
    var key = pair[0];
    print(key);
    var value = pair[1];

    if (key == "parent_directory"){
        parent_directory = value;
        print(parent_directory);
    }
    if (key == "thresh_min"){
        thresh_min = parseFloat(value);
        print(thresh_min);
    }
    if (key == "thresh_max"){
        thresh_max = parseFloat(value);
        print(thresh_max);
    }
}


// Function to check parent directory for multiple folders
function getFolders(parent_direct) {
    print(parent_direct);
    var folder = new File(parent_direct);
    print(folder);
    var allFiles = folder.list(); // Returns the list of folder names
    var folders = [];
    if (allFiles !== null) {
        print(allFiles);
        for (var i = 0; i < allFiles.length; i++) {
            var dirs = isDirectory(parent_directory + allFiles[i]);
            print(dirs);
            if (dirs == 1 ) {
                folders.push(parent_directory + allFiles[i]);
            }
        }
    }
    return folders;
}

// Check if the path has contents (indicating it's a directory)
function isDirectory(path) {
    var file = new File(path);
    var contents = file.list();
     if (contents !== null && contents.length > 0) {
        return 1;
     } else {
        return 0;
     }
}


// Function to list of files in the folder
function getFileList(raw_direct) {
    var folder = new File(raw_direct);
    var files = folder.list(); // Returns the list of file names
    return files;
}

// Create a new folder within the parent directory to store the processed images
function processFolder(direct) {
	var fileList = getFileList(direct);
    dir = direct.replace(/\/$/, "");
    var output_path = dir + "_MASKED/";
    var dir = new File(output_path);
    dir.mkdir();

// Process files in pairs based on the last number in their names
	for (var i = 0; i < fileList.length; i++) {

    // Get the last number from the current file
    	var currFile = (fileList[i]);
    	if (currFile.indexOf("ACTIN") !== -1) {

            // Get the image number of the actin file
    		var actin_number = String(currFile).match(/ACTIN(\d+)/);
    		var actin_number = actin_number[1];

            // Convert actin file to ImagePlus Object
            var actin_imp = IJ.openImage(direct + currFile);

            // Split the actin file
            var channels1 = ChannelSplitter.split(actin_imp);

            // Close non actin channels (channels 2 and 3)
            actin_imp.close();
            channels1[1].close();
            channels1[2].close();
            channels1[0].show();

            
            // Create mask
            // Apply thresholding to the first channel of actin channels
            IJ.run("8-bit");
            IJ.run("Threshold...", "min=" + thresh_min +" max=" + thresh_max);
            IJ.run("Auto Threshold", "method=Default B&W")
            IJ.run("Convert to Mask", "");
            IJ.run("Median...", "radius=3");
            IJ.run("Median...", "radius=3");
    		
                for (var j = 0; j < fileList.length; j++) {
                    var potentialDAPIFile = fileList[j];

                    // Check if the potential file is a DAPI file and has the same number
                    if ((potentialDAPIFile.indexOf("DAPI") !== -1)) {
                        var dapi_number = potentialDAPIFile.match(/DAPI(\d+)/);
                        var dapi_number = dapi_number[1];
                        
                        if(dapi_number === actin_number){
                        
                        // Convert DAPI file to ImagePlus Object
                        var dapi_imp = IJ.openImage(direct + potentialDAPIFile);

                        // Split the DAPI file, close the non DAPI channels
                            var channels2 = ChannelSplitter.split(dapi_imp);
                            channels2[2].show(); 
                            channels2[0].close();
                            channels2[1].close();
                            dapi_imp.close();

                        // Convert file to 8-bit 
                            IJ.run(channels2[2], "8-bit", "");

                        // Merge the new images
                            var merged = RGBStackMerge.mergeChannels([channels1[0], channels2[2]], false);
                            merged.show();
                            channels2[2].close();
                            channels1[0].close();
                            

                        // Save the processed images as TIFF
                            var finalPath = (output_path + currFile);
                            var modifiedPath = finalPath.replace(/\//g, "\\");
                            IJ.saveAs(merged, "Tiff", modifiedPath); // Save image
                            merged.close(); 
                            break;
                        }
                         }
                }
         }
         }
}


// Determine if the input is a folder or a file
var dir = isDirectory(parent_directory);

// If the input is a directory with subdirectories, process the subdirectories
if (dir == 1) {
    var raw_dirs = getFolders(parent_directory);
    print(raw_dirs);
    for (var i = 0; i < raw_dirs.length; i++) {
        print(raw_dirs[i]);
        print(raw_dirs[i] + "/");
        processFolder(raw_dirs[i] + "/");
    }
// If the input is a single folder, process the folder
    } else {

    processFolder(parent_directory);
}


