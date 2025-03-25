# parsers.py
import numpy as np
import re
import os

from SmallCellMTPTraining.templates import properties as properties


def extractAllNumbersFromString(sequence: str) -> list:
    return [float(x) for x in re.findall(r"-?\d\d*\.?\d*", sequence)]


def parseQEOutput(fileName: str, convertToAngRy=True) -> dict:
    energyConversion = properties.energyConversion
    distanceConversion = properties.distanceConversion
    if not convertToAngRy:
        energyConversion = distanceConversion = 1

    with open(fileName, "r") as txtfile:
        content = txtfile.read()

        # Check for completion:
        if re.search(r"stopping...", content):
            print(fileName + " has failed!!")
            return {"jobComplete": False}

        # Get the configuration energy
        energy = (
            float(re.findall(r"(?:.*!\s*total energy\s*=\s*)(\S+)", content)[0])
            * energyConversion
        )  # Extracts the energies

        # Get the forces
        atomTypeAndForceStrings = re.findall(
            r".*atom.*type .*\n", content
        )  # Extracts the atom force line
        atomTypeAndForceFloats = np.array(
            [extractAllNumbersFromString(x) for x in atomTypeAndForceStrings]
        )  # Extracts the numbers from the atom force line into an numpy array
        atomIDs = atomTypeAndForceFloats[:, 0]
        atomTypes = atomTypeAndForceFloats[:, 1].astype(int)
        atomForces = (
            atomTypeAndForceFloats[:, 2:] * energyConversion / distanceConversion
        )

        # Get the lattice parameter
        latticeParameter = extractAllNumbersFromString(
            re.findall(r"celldm\(1\)=\s*\d*\.?\d*", content)[0]
        )[1]

        # Get the atom position vectors
        positionVectorsStrings = re.findall(
            r"\d\s*[A-Za-z]+\s+tau\(\s+\d+\) = \(\s*-?\d*\.?\d*\s*-?\d*\.?\d*\s*-?\d*\.?\d*  \)",
            content,
        )
        # print(positionVectorsStrings)
        positionVectorsFloats = np.array(
            [extractAllNumbersFromString(x) for x in positionVectorsStrings]
        )[:, 2:]
        positionVectors = positionVectorsFloats * latticeParameter * distanceConversion

        # Get the supercell vectors
        superCellVectorsStrings = re.findall(
            r".*a\(\d\).*\n", content
        )  # Extracts the supercell lines
        superCellVectorsFloats = np.array(
            [extractAllNumbersFromString(x) for x in superCellVectorsStrings]
        )
        superCellVectors = (
            superCellVectorsFloats[:, 1:] * latticeParameter * distanceConversion
        )

        # Calculate the Volume
        A2vA3_1 = (
            superCellVectors[1][1] * superCellVectors[2][2]
            - superCellVectors[1][2] * superCellVectors[2][1]
        )
        A2vA3_2 = (
            superCellVectors[1][2] * superCellVectors[2][0]
            - superCellVectors[1][0] * superCellVectors[2][2]
        )
        A2vA3_3 = (
            superCellVectors[1][0] * superCellVectors[2][1]
            - superCellVectors[1][1] * superCellVectors[2][0]
        )
        V = (
            superCellVectors[0][0] * A2vA3_1
            + superCellVectors[0][1] * A2vA3_2
            + superCellVectors[0][2] * A2vA3_3
        )

        txtfile.seek(0)
        fileLines = txtfile.readlines()
        index = 0
        timeIndex = 0
        for i, line in enumerate(fileLines):
            if re.search(r"total   stress", line):
                index = i
            if re.search(r"     Parallel routines", line):
                timeIndex = i + 2

        # Get the stress vectors
        v1 = np.array(fileLines[index + 1].split(), dtype=float)  # Read stresses
        v2 = np.array(fileLines[index + 2].split(), dtype=float)
        v3 = np.array(fileLines[index + 3].split(), dtype=float)
        stressVectors = (
            np.array([v1, v2, v3])[:, :3] * V * energyConversion / distanceConversion**3
        )

        # Get the PlusStress
        virialStressFloats = [
            stressVectors[0][0],
            stressVectors[1][1],
            stressVectors[2][2],
            stressVectors[1][2],
            stressVectors[0][2],
            stressVectors[0][1],
        ]
        # print(fileName, energy)
        # Get the pressures
        pressure = extractAllNumbersFromString(
            re.findall(r"P=\s*-?\d\d*.?\d*", content)[0]
        )[0]

        cpuTimeMatchingPattern = r"""
        PWSCF\s*:\s*  # Match "PWSCF:" with optional whitespace
        (?:           # Start of non-capturing group for days
        (\d+)d\s*   # Match days (optional) with "d" and optional whitespace
        )?            # End of non-capturing group for days
        (?:           # Start of non-capturing group for hours
        (\d+)h\s*   # Match hours (optional) with "h" and optional whitespace
        )?            # End of non-capturing group for hours
        (?:           # Start of non-capturing group for minutes
        (\d+)m\s*   # Match minutes (optional) with "m" and optional whitespace 
        )?            # End of non-capturing group for minutes
        (?:           # Start of non-capturing group for seconds
        (\d+(?:\.\d+)?)s   # Match seconds (optional) with "s" and optional decimal part
        )?            # End of non-capturing group for seconds
        \s*CPU       # Match "CPU" with optional whitespace
        """
        all_matches = re.findall(cpuTimeMatchingPattern, content, re.VERBOSE)

        for match in all_matches:
            days, hours, minutes, seconds = match
            total_seconds = (
                (int(days or 0) * 24 * 60 * 60)
                + (int(hours or 0) * 60 * 60)
                + (int(minutes or 0) * 60)
                + float(seconds or 0)
            )

        return {
            "jobComplete": True,
            "energy": energy,
            "atomIDs": atomIDs,
            "atomTypes": atomTypes,
            "atomPositions": positionVectors,
            "atomForces": atomForces,
            "superCell": superCellVectors,
            "stressVectors": stressVectors,
            "virialStresses": virialStressFloats,
            "pressure": pressure,
            "cpuTimeSpent": total_seconds,
        }


def parseMTPConfig(startIndex: int, fileLines: list, convertFromAngRy=True) -> dict:
    energyConversion = 1 / 13.605693012183622
    distanceConversion = 1 / 0.5291772105638411
    if not convertFromAngRy:
        energyConversion = distanceConversion = 1

    numAtoms = int(fileLines[startIndex + 2].split()[0])

    energy = float(fileLines[startIndex + 9 + numAtoms].split()[0]) * energyConversion

    v1 = np.array(fileLines[startIndex + 4].split(), dtype=float)  # Read supercell
    v2 = np.array(fileLines[startIndex + 5].split(), dtype=float)
    v3 = np.array(fileLines[startIndex + 6].split(), dtype=float)
    superCellVectors = np.array([v1, v2, v3]) * distanceConversion

    infoArray = np.zeros((numAtoms, 8))

    for j in range(numAtoms):
        infoArray[j] = np.array(fileLines[startIndex + 8 + j].split(), dtype=float)

    atomIDs = infoArray[:, 0]
    atomTypes = infoArray[:, 1].astype(int)
    atomPositions = infoArray[:, 2:5] * distanceConversion
    atomForces = infoArray[:, 5:] * energyConversion / distanceConversion

    virialStresses = np.array(
        fileLines[startIndex + 11 + numAtoms].split(), dtype=float
    )

    return {
        "numAtoms": numAtoms,
        "energy": energy,
        "atomIDs": atomIDs,
        "atomTypes": atomTypes,
        "atomPositions": atomPositions,
        "atomForces": atomForces,
        "superCell": superCellVectors,
        "virialStresses": virialStresses,
    }


def parseMTPConfigsFile(filename: str, convertFromAngRy=True) -> list:
    properties = []
    with open(filename, "r") as txtfile:
        fileLines = txtfile.readlines()
        indicies = np.where(np.array(fileLines) == "BEGIN_CFG\n")[
            0
        ]  # Seach for indicies which match the beginning of a configuration
        for i in indicies:
            properties.append(parseMTPConfig(i, fileLines, convertFromAngRy))
    return properties


def parseAllQEInDirectory(dirName: str, convertFromAngRy=True) -> list:
    if not os.path.isdir(dirName):
        raise Exception("The output directory does not exist!")
    qeOutputs = []
    for filename in sorted(os.listdir(dirName)):
        if filename.endswith(".out"):
            output = parseQEOutput(os.path.join(dirName, filename), convertFromAngRy)
            if output["jobComplete"] == False:
                os.remove(os.path.abspath(filename))
            else:
                qeOutputs.append(output)

    return qeOutputs


def parsePartialMTPConfig(startIndex: int, fileLines: list) -> dict:
    numAtoms = int(fileLines[startIndex + 2].split()[0])

    v1 = np.array(fileLines[startIndex + 4].split(), dtype=float)  # Read supercell
    v2 = np.array(fileLines[startIndex + 5].split(), dtype=float)
    v3 = np.array(fileLines[startIndex + 6].split(), dtype=float)
    superCellVectors = np.array([v1, v2, v3])

    infoArray = np.zeros((numAtoms, 5))

    for j in range(numAtoms):
        infoArray[j] = np.array(fileLines[startIndex + 8 + j].split(), dtype=float)[:5]

    atomIDs = infoArray[:, 0]
    atomTypes = infoArray[:, 1].astype(int)
    atomPositions = infoArray[:, 2:5]

    # Extract MV_grade using regex
    mv_grade_line_index = startIndex + 8 + numAtoms
    mv_grade_match = re.search(
        r"Feature\s+MV_grade\s+([\d\.-]+)", fileLines[mv_grade_line_index]
    )
    mv_grade = float(mv_grade_match.group(1)) if mv_grade_match else None

    return {
        "numAtoms": numAtoms,
        "atomIDs": atomIDs,
        "atomTypes": atomTypes,
        "atomPositions": atomPositions,
        "superCell": superCellVectors,
        "MV_grade": mv_grade,
    }


def parsePartialMTPConfigsFile(filename: str) -> list:
    properties = []
    with open(filename, "r") as txtfile:
        fileLines = txtfile.readlines()
        indicies = np.where(np.array(fileLines) == "BEGIN_CFG\n")[
            0
        ]  # Seach for indicies which match the beginning of a configuration
        for i in indicies:
            properties.append(parsePartialMTPConfig(i, fileLines))
    return properties


def parseTimeFile(filename: str) -> float:
    try:
        with open(filename, "r") as f:
            content = f.read()

        # Check if there is an error in the time file
        error_match = re.match(r"Command exited with non-zero status (\d+)", content)
        if error_match:
            status_code = int(error_match.group(1))
            if status_code not in (0, 9):
                raise RuntimeError(
                    f"Command exited with unexpected status: {status_code}"
                )

            # If status is 0 or 9, we can proceeed
            time_str = content.replace(error_match.group(0), "").strip()
            if not time_str:  # Check if now is empty
                raise ValueError("Time value not found after removing error message.")
            return float(time_str)
        else:  # If no error, just convert to float
            content = content.strip()
            if not content:
                raise ValueError("File is empty")
            return float(content)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{filename}'")


def parseMDTime(filename: str) -> float:
    """Extracts the time from a file with the given filename."""

    with open(filename, "r") as f:
        lines = f.readlines()

    pattern = r"Total wall time:\s*(\d+):(\d+):(\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            total_seconds = (hours * 60 * 60) + (minutes * 60) + seconds
            return total_seconds


if __name__ == "__main__":

    # print(parsePartialMTPConfigsFile("preselected.cfg.0"))
    print(parseQEOutput("1_0_0_0.out"))
# print(
#     parseMTPConfigsFile(
#         "/global/home/hpc5146/Projects/KTraining/temporaryFiles/train.cfg", False
#     )
# )
