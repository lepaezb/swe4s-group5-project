// Establish environment to run the fiji macro
var ImagePlus = org.imagej.ImagePlus;
importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.process.ImageProcessor);
importClass(Packages.ij.plugin.ChannelSplitter);
importClass(Packages.ij.plugin.RGBStackMerge);
importClass(java.io.File);

// Parse the arguments provided from the python input 
    // var args = getArgument();
    // var argArray = split(args, ",");

    // for (i=0; i<splitArgs.length; i++) {
    //     keyValue = splitArgs[i].split("=");
    //     if (keyValue[0] == "raw_directory"){
    //         var raw_directory = keyValue[1];
    //     }
    //     if (keyValue[0] == "output_path"){
    //         var output_path = keyValue[1];
    //     }
    // if (keyValue[0] == "thresh_min"){
    //     var thresh_min = parseInt(keyValue[1], 10);
    // }
    // if (keyValue[0] == "thresh_max"){
    //     var thresh_max = parseInt(keyValue[1], 10);
    // }
//}

var parent_directory = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/TEST_DIR/";

// Function to check parent directory for multiple folders
function getFolders(parent_direct) {
    var folder = new File(parent_direct);
    var allFiles = folder.list(); // Returns the list of folder names
    var folders = [];
    if (allFiles !== null) {
        for (var i = 0; i < allFiles.length; i++) {
            print(allFiles[i]);
            var dirs = isDirectory(parent_directory + allFiles[i]);
            if (dirs == 1 ) {
                folders.push(parent_directory + allFiles[i]);
            }
        }
    }
    return folders;
}

function isDirectory(path) {
    var file = new File(path);
    // Check if the path has contents (indicating it's a directory)
    var contents = file.list();
     if (contents !== null && contents.length > 0) {
        return 1;
     } else {
        return 0;
     }
}


// Function to list files in the folder
function getFileList(raw_direct) {
    var folder = new File(raw_direct);
    var files = folder.list(); // Returns the list of file names
    return files;
}


function processFolder(direct) {
	var fileList = getFileList(direct);
    var output_path = direct + "MASKED/";
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
            // Apply thresholding to the first channel of actinChannels
            IJ.run("8-bit");
            IJ.run("Threshold...", "min=5500 max=10500");
            IJ.run("Auto Threshold", "method=Default B&W")
            IJ.run("Convert to Mask", "");
            // White borders on a black background
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
                            dapi_imp.close();
                            channels2[0].close();
                            channels2[1].close();
                            channels2[2].show();
                            

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
print(dir);
if (dir == 1) {
    print("in loop");
    var raw_dirs = getFolders(parent_directory);
    print(raw_dirs);
    for (var i = 0; i < raw_dirs.length; i++) {
        processFolder(raw_dirs[i] + "/");
    }
    } else {

    processFolder(parent_directory);
}

