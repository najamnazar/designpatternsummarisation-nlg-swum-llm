package dps_nlg.utils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.github.javaparser.ast.NodeList;

/**
 * Utility class providing helper methods for NLG summary generation.
 * <p>
 * This class contains static utility methods for extracting and organizing
 * data from parsed Java structures, including class/interface details,
 * method information, and design pattern relationships. It supports the
 * NLG summarization process by structuring data for sentence generation.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Extract class and interface metadata from parsed structures</li>
 *   <li>Extract method details including signatures and annotations</li>
 *   <li>Organize design pattern information into multi-valued maps</li>
 *   <li>Support relationship extraction for inheritance and implementation</li>
 *   <li>Provide file and string manipulation utilities</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class Utils {
    /**
     * Gets the file extension from a file.
     * 
     * @param f the file to extract extension from
     * @return the file extension in lowercase, or empty string if none
     */
    public static String getExtension(File f) {
        String ext = null;
        String s = f.getName();
        int i = s.lastIndexOf('.');

        if (i > 0 && i < s.length() - 1) {
            ext = s.substring(i + 1).toLowerCase();
        }
        if (ext == null) {
            ext = "";
        }
        return ext;
    }

    /**
     * Extracts the base name of a file (without extension).
     * 
     * @param fileName the file name to process
     * @return the file name without extension
     */
    public static String getBaseName(String fileName) {
        int index = fileName.lastIndexOf('.');
        if (index == -1) {
            return fileName;
        } else {
            return fileName.substring(0, index);
        }
    }

    // get by dot, used for string separation

    /**
     * @param parsedString
     * @param part         which part of the splited string to be returned. 1 for
     *                     method and 2 for class name when use for caller, 1 for
     *                     class when use for param
     * @return
     */
    public static String splitByDot(String parsedString, int part) {
        if (parsedString.contains(".")) {
            String[] parsedStringList = parsedString.split("\\.", -1);
            return parsedStringList[parsedStringList.length - part];
        } else {
            return parsedString;
        }
    }

    public static String removeSpecialCharacter(String parsedString) {
        return parsedString.replaceAll("[^a-zA-Z]+", "");
    }

    // referenced from Java Callgraph
    /**
     * 判断是否需要过滤
     * 
     * @param s
     * @param skipPatterns
     * @return
     */
    public static boolean shouldSkip(String s, List<Pattern> skipPatterns) {
        for (Pattern skipPattern : skipPatterns) {
            if (skipPattern.matcher(s).matches()) {
                return true;
            }
        }

        return false;
    }

    // referenced from Java Callgraph
    /**
     * 辅助函数， 根据后缀名筛选文件
     * 
     * @param suffix
     * @param path
     * @return
     */
    public static List<String> getFilesBySuffixInPath(String suffix, String path) throws IOException {
        return Files.find(Paths.get(path), Integer.MAX_VALUE, (filePath, fileAttr) -> fileAttr.isRegularFile())
                .filter(f -> f.toString().toLowerCase().endsWith(suffix))
                .map(f -> f.toString()).collect(Collectors.toList());
    }

    // referenced from Java Callgraph
    /**
     * 获取目录下以某个字符串为结尾的文件列表
     * 
     * @param suffix
     * @param paths
     * @return
     */
    public static List<String> getFilesBySuffixInPaths(String suffix, List<String> paths) throws IOException {
        List<String> files = new ArrayList<>();
        for (String path : paths) {
            files.addAll(getFilesBySuffixInPath(suffix, path));
        }
        return files;
    }

    // referenced from Java Callgraph
    /**
     * 单元素转换成列表
     * 
     * @param <T>
     * @param object
     * @return
     */
    public static <T> List<T> makeListFromOneElement(T object) {
        ArrayList<T> list = new ArrayList<>();
        if (object != null) {
            list.add(object);
        }
        return list;
    }

    public static ArrayList<String> nodeListToArrayList(NodeList nodeList) {
        ArrayList<String> arrayList = new ArrayList<String>();
        for (Object node : nodeList) {
            arrayList.add(node.toString().strip());
        }
        return arrayList;
    }

    public static String convertToPlainText(String text) {
        if (text.contains("_"))
            text = text.replace("_", " ");
        return text;
    }

    public static String[] splitByCamelCase(String text) {
        String regex = "(?<=\\p{Ll})(?=\\p{Lu})";

        return text.split(regex);
    }

    public static HashMap getProjectDetails(HashMap<String, HashMap> fileDetails) {
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet())
            if (!fileEntry.getKey().equals("design_pattern"))
                return fileEntry.getValue();
        return new HashMap<>();
    }

    public static ArrayList<HashMap> getMethodDetails(HashMap classDetails) {
        return (ArrayList<HashMap>) classDetails.get("METHODDETAIL");
    }

    public static ArrayList<HashMap> getClassOrInterfaceDetails(HashMap classDetails) {
        return (ArrayList<HashMap>) classDetails.get("CLASSORINTERFACEDETAIL");
    }

    public static ArrayList<HashMap> getFieldDetails(HashMap classDetails) {
        return (ArrayList<HashMap>) classDetails.get("FIELDDETAIL");
    }

    public static ArrayList<HashMap> getConstructorDetails(HashMap classDetails) {
        return (ArrayList<HashMap>) classDetails.get("CONSTRUCTORDETAIL");
    }

    public static ArrayList<HashMap> getVariableDetails(HashMap classDetails) {
        return (ArrayList<HashMap>) classDetails.get("VARIABLEDETAIL");
    }

    public static ArrayList<String> getImplementsFrom(HashMap classDetail) {
        return (ArrayList<String>) classDetail.get("IMPLEMENTSFROM");
    }

    public static ArrayList<String> getExtendsFrom(HashMap classDetail) {
        return (ArrayList<String>) classDetail.get("EXTENDSFROM");
    }

    public static String getClassName(HashMap classDetail) {
        return (String) classDetail.get("CLASSNAME");
    }

    public static String getMethodReturnType(HashMap methodDetail) {
        return (String) methodDetail.get("METHODRETURNTYPE");
    }

    public static String getMethodName(HashMap methodDetail) {
        return (String) methodDetail.get("METHODNAME");
    }

    public static ArrayList<String> getMethodParameterAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();
        for (HashMap methodParameter : (ArrayList<HashMap>) methodDetail.get("METHODPARAMETER")) {
            String methodParameterType = (String) methodParameter.get("PARAMETERTYPE");
            String methodParameterName = (String) methodParameter.get("PARAMETERNAME");

            String resultString = methodParameterType + " parameter of " + methodParameterName;
            resultArray.add(resultString);
        }
        return resultArray;
    }

    public static ArrayList<HashMap> getParameters(NodeList parameters) {
        ArrayList<HashMap> resultArray = new ArrayList<>();
        for (Object parameter : parameters) {
            String[] parameterString = parameter.toString().split(" ");
            if (parameterString.length < 1)
                continue;
            HashMap resultMap = new HashMap<>();
            resultMap.put("PARAMETERTYPE", parameterString[0]);
            resultMap.put("PARAMETERNAME", parameterString[1]);
            resultArray.add(resultMap);
        }
        return resultArray;
    }

    public static boolean isMethodOverride(HashMap methodDetail) {
        ArrayList<String> methodOverride = (ArrayList) methodDetail.get("METHODOVERRIDE");
        return methodOverride.contains("@Override");
    }

    public static boolean isInterfaceOrNot(HashMap classDetail) {
        return (boolean) classDetail.get("ISINTERFACEORNOT");
    }

    // public static String getVariableElementType(HashMap variableDetail) {
    // return (String) variableDetail.get("VARIABLEELEMENTTYPE");
    // }

    public static String getFieldDataType(HashMap fieldDetail) {
        return (String) fieldDetail.get("FIELDDATATYPE");
    }

    public static ArrayList<String> getFieldModifierType(HashMap fieldDetail) {
        return (ArrayList<String>) fieldDetail.get("FIELDMODIFIERTYPE");
    }

    public static ArrayList<String> getMethodModifierType(HashMap methodDetail) {
        return (ArrayList<String>) methodDetail.get("METHODMODIFIERTYPE");
    }

    public static ArrayList<String> getConstructorModifier(HashMap constructorDetail) {
        return (ArrayList<String>) constructorDetail.get("CONSTRUCTORMODIFIER");
    }

    public static ArrayList<String> getClassModifierType(HashMap classDetail) {
        return (ArrayList<String>) classDetail.get("CLASSMODIFIERTYPE");
    }

    public static ArrayList<HashMap> getConstructorParameters(HashMap constructorDetail) {
        return (ArrayList<HashMap>) constructorDetail.get("CONSTRUCTORPARAMETER");
    }

    public static ArrayList<String> getIncomingMethodAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();
        for (HashMap incomingMethod : (ArrayList<HashMap>) methodDetail.get("INCOMINGMETHOD")) {
            String incomingMethodClass = (String) incomingMethod.get("CALLEDCLASS");
            String incomingMethodName = (String) incomingMethod.get("CALLEDMETHODNAME");

            String resultString = incomingMethodName + " method of " + incomingMethodClass;
            if (!resultArray.contains(resultString)) {
                resultArray.add(resultString);
            }
        }
        return resultArray;
    }

    public static ArrayList<String> getOutgoingMethodAsText(HashMap methodDetail) {
        ArrayList<String> resultArray = new ArrayList<>();

        for (HashMap outgoingMethod : (ArrayList<HashMap>) methodDetail.get("OUTGOINGMETHOD")) {
            String outgoingMethodClass = (String) outgoingMethod.get("CALLEECLASS");
            String outgoingMethodName = (String) outgoingMethod.get("CALLEEMETHODNAME");

            String resultString = outgoingMethodName + " method of " + outgoingMethodClass;
            if (!resultArray.contains(resultString)) {
                resultArray.add(resultString);
            }
        }
        return resultArray;
    }

    public static ArrayList<HashMap> getMethodParameters(HashMap methodDetail) {
        return (ArrayList<HashMap>) methodDetail.get("METHODPARAMETER");
    }

    // Compares methods of parent and current class.
    // Check if name, parameters and return type are matching and overridden,
    // and specify the equivalence as a string
    public static ArrayList<String> checkMethodOverride(HashMap currentClass, HashMap parentClass,
            String methodOfParentString) {
        ArrayList<String> overrideMethodArray = new ArrayList<>();
        for (HashMap methodDetail : Utils.getMethodDetails(currentClass)) {
            String currentMethodReturnType = Utils.getMethodReturnType(methodDetail);
            String currentMethodName = Utils.getMethodName(methodDetail);
            ArrayList<String> currentMethodParametersAsText = Utils.getMethodParameterAsText(methodDetail);

            if (!Utils.isMethodOverride(methodDetail))
                continue;

            for (HashMap parentMethodDetail : Utils.getMethodDetails(parentClass)) {
                String parentMethodReturnType = Utils.getMethodReturnType(parentMethodDetail);
                String parentMethodName = Utils.getMethodName(parentMethodDetail);
                ArrayList<String> parentMethodParametersAsText = Utils
                        .getMethodParameterAsText(parentMethodDetail);

                if (currentMethodName.equals(parentMethodName)
                        && currentMethodParametersAsText.equals(parentMethodParametersAsText)
                        && currentMethodReturnType.equals(parentMethodReturnType)) {
                    overrideMethodArray.add(parentMethodName + methodOfParentString);
                }
            }
        }
        return overrideMethodArray;
    }

    public static ArrayList<HashMap> getOutgoingMethod(HashMap methodDetail) {
        return (ArrayList<HashMap>) methodDetail.get("OUTGOINGMETHOD");
    }

    public static ArrayList<HashMap> getIncomingMethod(HashMap methodDetail) {
        return (ArrayList<HashMap>) methodDetail.get("INCOMINGMETHOD");
    }

    public static String getIncomingMethodClass(HashMap incomingMethod) {
        return (String) incomingMethod.getOrDefault("CALLEDCLASS", "");
    }

    public static String getIncomingMethodName(HashMap incomingMethod) {
        return (String) incomingMethod.getOrDefault("CALLEDMETHODNAME", "");
    }

    public static String getOutgoingMethodClass(HashMap outgoingMethod) {
        return (String) outgoingMethod.getOrDefault("CALLEECLASS", "");
    }

    public static String getOutgoingMethodName(HashMap outgoingMethod) {
        return (String) outgoingMethod.getOrDefault("CALLEEMETHODNAME", "");
    }

    public static String getParameterType(HashMap parameters) {
        return (String) parameters.getOrDefault("PARAMETERTYPE", "");
    }

    public static String getParameterName(HashMap parameters) {
        return (String) parameters.getOrDefault("PARAMETERNAME", "");
    }

    public static String getMethodNameFromMatchingParameterType(HashMap methodDetail, String className) {
        for (HashMap parameter : (ArrayList<HashMap>) Utils.getMethodParameters(methodDetail)) {
            if (Utils.getParameterType(parameter).equals(className)) {
                return (String) Utils.getMethodName(methodDetail);
            }
        }
        return "";
    }

    public static String getMethodNameFromMatchingReturnType(HashMap methodDetail, String className) {

        if (className.equals(Utils.getMethodReturnType(methodDetail)))
            return Utils.getMethodName(methodDetail);
        return "";
    }

    public static ArrayList<String> getMethodNameFromMatchingIncomingMethod(HashMap methodDetail, String className,
            String originClass) {

        ArrayList<String> resultArrayList = new ArrayList<>();
        for (HashMap incomingMethod : (ArrayList<HashMap>) Utils.getIncomingMethod(methodDetail)) {
            if (className.equals(Utils.getIncomingMethodClass(incomingMethod))) {
                resultArrayList.add(Utils.getMethodName(methodDetail) + " method of " + originClass);
            }
        }
        return resultArrayList;
    }

}

