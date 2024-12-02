// edit this macro so it runs in fiji headless 
// Establish environment to run the fiji macro
var ImagePlus = org.imagej.ImagePlus;
importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.process.ImageProcessor);
importClass(Packages.ij.plugin.ChannelSplitter);
importClass(Packages.ij.plugin.RGBStackMerge);

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

var raw_directory = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/TEST_DIR/TEST_1/"
var output_path = "C:/Users/laure/OneDrive - UCB-O365/ROTATION1/DDX6 Pilot/Reps 1-3 LO/TEST_DIR/TEST_1/MASKED/"

// Function to list files in the folder
function getFileList(raw_directory) {
    importClass(Packages.java.io.File);
    var folder = new File(raw_directory);
    var files = folder.list(); // Returns the list of file names
    return files;
}

// Split channels 
// function splitChannels(imp) {
//   imp = IJ.getImage(imp);
//   var channels = ChannelSplitter.split(imp);

//   for (var i = 0; i < channels.length; i++) {
//     var channel = channels[i];
//     channel.show();
//   }
// }


function processFolder(raw_directory, output_path) {
	var fileList = getFileList(raw_directory);

// Process files in pairs based on the last number in their names
	for (var i = 0; i < fileList.length; i++) {

    // Get the last number from the current file
    	var currFile = (fileList[i]);
    	if (currFile.indexOf("ACTIN") !== -1) {

            // Get the image number of the actin file
    		var actin_number = String(currFile).match(/ACTIN(\d+)/);
    		var actin_number = actin_number[1];

            // Convert actin file to ImagePlus Object
            var actin_imp = IJ.openImage(raw_directory + currFile);

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
            IJ.run("Convert to Mask", "");
            IJ.run("Invert LUT"); // White borders on a black background
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
                        var dapi_imp = IJ.openImage(raw_directory + potentialDAPIFile);

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

                        // Save the processed images as TIFF
                            IJ.saveAs("Tiff", output_path + currFile); // Save image
                            merged.close();

                        // Call processFile with the found pair: processFile(raw_directory, output_path, currFile, potentialDAPIFile);
                            break;
                        }
                         }
                }
         }
         }
}

processFolder(raw_directory, output_path)