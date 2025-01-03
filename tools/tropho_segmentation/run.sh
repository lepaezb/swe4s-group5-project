# tropho_segmentation_run.sh

Help() {
    # Help message 
    echo "Function descriptions are as follows"
    echo 
    echo "Syntax: scriptTemplate [-d|c|f|m|M|h]"
    echo "Options:"
    echo "d: parent directory path to retrieve files of interest (required)"
    echo "f: location of fiji on local machine (required)" 
    echo "c: channel number to segment"
    echo "m: minimum threshold value"
    echo "M: maximum threshold value"
    echo "h: type h for help message"
    exit 0
} 

# Options for variables to enter
while getopts "d:c:f:m:M:h" flag
do
    case "${flag}" in
    #list the variables of input; optarg 
        d) parent_directory=${OPTARG};;
        c) channel_number=${OPTARG};;
        f) fiji_path=${OPTARG};;
        m) thresh_min=${OPTARG};;
        M) thresh_max=${OPTARG};;
        h) #HELP 
            Help
            exit;;
        *) #invalid option
            echo "Error: invalid variable"
            exit 1;;
    esac
done


# Check that required variables have been inputted
if [[ -z "$parent_directory" || -z "$fiji_path" ]]; then
    echo "Error: missing necessary argument."
    Help
fi


# Return the user inputs
echo "Parent directory: $parent_directory"
echo "Fiji location on local machine: $fiji_path"

if [[ -n "$channel_number" ]]; then
    echo "Channel for segmentation in cellpose: $channel_number"
else
    channel_number=2
    echo "Channel for segmentation in cellpose: 2"
fi


if [[ -n "$thresh_min" ]]; then
    echo "Minimum threshold value: $thresh_min"
else
    thresh_min=5500
    echo "Minimum threshold value: 5500"
fi

if [[ -n "$thresh_max" ]]; then
    echo "Maximum threshold value: $thresh_max"
else 
    thresh_max=10500
    echo "Maximum threshold value: 10500"
fi  


# Check that parent directory exists
if [[ ! -d "$parent_directory" ]] ; then
    echo "'$parent_directory' is not a directory."
    exit 1
fi

# Check that parent directory contains files 
if [[ ! $(ls -A "$parent_directory") ]] ; then
    echo "No files found in "$parent_directory"."
    exit 1
fi


# Function to check if a variable is an integer
int_check() {
    local s="$1"
    if [[ "$s" =~ ^[0-9]+$ ]]; then
        return 0  # Success, it's an integer
    else
        return 1  # Failure, it's not an integer
    fi
}

# Check that channel_number variable is an integer
if [[ ! -z "$channel_number" ]] ; then
    if int_check "$channel_number" ; then
        # Check that channel number is 0,1,2,3
        if [ "$channel_number" -lt 0 ] || [ "$channel_number" -gt 3 ] ; then
            echo "Error: channel number must be 0, 1, 2, or 3."
            exit 1
        fi

        # Converts to integer
        channel_number=$(echo "$channel_number" | sed 's/[^0-9]//g')

        # Check if conversion worked
        if [ ! -n "$channel_number" ] ; then
            echo "Conversion of "$channel_number" to integer failed."
            exit 1
        else
            echo "Conversion of "$channel_number" to integer successful."
        fi
    fi
fi

# Check that thresh_min and thresh_max are integers
if [[ ! -z "$thresh_min" ]] ; then
    if int_check "$thresh_min" ; then
        # Converts to integer
        thresh_min=$(echo "$thresh_min" | sed 's/[^0-9]//g')

        # Check if conversion worked
        if [ ! -n "$thresh_min" ] ; then
            echo "Conversion of "$thresh_min" to integer failed."
            exit 1
        else
            echo "Conversion of "$thresh_min" to integer successful."
        fi
    fi
fi

if [[ ! -z "$thresh_max" ]] ; then
    if int_check "$thresh_max" ; then
        # Converts to integer
        thresh_max=$(echo "$thresh_max" | sed 's/[^0-9]//g')

        # Check if conversion worked
        if [ ! -n "$thresh_max" ] ; then
            echo "Conversion of "$thresh_max" to integer failed."
            exit 1
        else
            echo "Conversion of "$thresh_max" to integer successful."
        fi
    fi
fi


# Pass arguments to segmentation_cellpose.py 
python segmentation_cellpose.py --parent_directory "$parent_directory" \
--channel_number "$channel_number" \
--fiji_path "$fiji_path" \
--thresh_min "$thresh_min" \
--thresh_max "$thresh_max"






