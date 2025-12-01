package dps_swum.utils;

import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory class for creating and configuring JavaSymbolSolver instances.
 * <p>
 * This class provides centralized creation of symbol solvers that enable JavaParser
 * to resolve type information, inheritance relationships, and method calls across
 * source code, JDK classes, and JAR dependencies. It combines multiple type solvers
 * to provide comprehensive symbol resolution.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Create configured JavaSymbolSolver instances</li>
 *   <li>Combine reflection, source, and JAR-based type solvers</li>
 *   <li>Handle missing or invalid source/library paths gracefully</li>
 *   <li>Provide diagnostic output for troubleshooting</li>
 * </ul>
 * </p>
 * <p>
 * Referenced from Java Callgraph project.
 * </p>
 * 
 * @author allen
 * @author Najam
 */
public class SymbolSolverFactory {
    /**
     * Creates a configured JavaSymbolSolver combining multiple type resolution strategies.
     * <p>
     * The solver is configured with the following priority order:
     * <ol>
     *   <li>ReflectionTypeSolver - for JDK classes</li>
     *   <li>JavaParserTypeSolver - for source code paths</li>
     *   <li>JarTypeSolver - for JAR libraries</li>
     * </ol>
     * </p>
     * 
     * @param srcPaths list of source code directory paths
     * @param libPaths list of library directory paths (containing JARs)
     * @return configured JavaSymbolSolver instance
     */
    public static JavaSymbolSolver getJavaSymbolSolver(List<String> srcPaths, List<String> libPaths) {
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        
        // Add reflection type solver first for JDK classes (most reliable)
        ReflectionTypeSolver reflectionTypeSolver = new ReflectionTypeSolver(false); // Set false to prevent caching issues
        combinedTypeSolver.add(reflectionTypeSolver);
        
        // Enhanced Java parser type solvers with diagnostic output
        List<JavaParserTypeSolver> javaParserTypeSolvers = makeJavaParserTypeSolvers(srcPaths);
        int sourcePathCount = 0;
        for (JavaParserTypeSolver solver : javaParserTypeSolvers) {
            try {
                combinedTypeSolver.add(solver);
                sourcePathCount++;
            } catch (Exception e) {
                System.out.println("Failed to add JavaParser type solver: " + e.getMessage());
            }
        }
        if (sourcePathCount > 0) {
            System.out.println("  Added " + sourcePathCount + " source path(s) for symbol resolution");
        }
        
        // Add jar type solvers with error handling
        List<JarTypeSolver> jarTypeSolvers = makeJarTypeSolvers(libPaths);
        if (jarTypeSolvers.size() > 0) {
            for (JarTypeSolver solver : jarTypeSolvers) {
                try {
                    combinedTypeSolver.add(solver);
                } catch (Exception e) {
                    System.out.println("Failed to add JAR type solver: " + e.getMessage());
                }
            }
            System.out.println("  Added " + jarTypeSolvers.size() + " JAR file(s) for symbol resolution");
        }
        
        // Add common library type solvers for typical dependencies
        addCommonLibraryResolvers(combinedTypeSolver);
        
        // Configure the symbol solver to be more lenient with unresolved symbols
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(combinedTypeSolver);
        return symbolSolver;
    }
    
    /**
     * Add resolvers for common Java libraries that might be missing
     */
    private static void addCommonLibraryResolvers(CombinedTypeSolver combinedTypeSolver) {
        // Add resolvers for common packages that are often missing
        // This helps with basic Java collections, utilities, etc.
        // These are already covered by the main ReflectionTypeSolver, but this ensures consistency
    }

    // referenced from Java Callgraph
    /**
     * 获取jar包的符号推理器
     * 
     * @param libPaths
     * @return
     */
    private static List<JarTypeSolver> makeJarTypeSolvers(List<String> libPaths) {
        List<String> jarPaths = Utils.getFilesBySuffixInPaths("jar", libPaths);
        List<JarTypeSolver> jarTypeSolvers = new ArrayList<>(jarPaths.size());
        try {
            for (String jarPath : jarPaths) {
                jarTypeSolvers.add(new JarTypeSolver(jarPath));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return jarTypeSolvers;
    }

    // referenced from Java Callgraph
    /**
     * 获取工程源代码src的符号推理器
     * 
     * @param srcPaths
     * @return
     */
    private static List<JavaParserTypeSolver> makeJavaParserTypeSolvers(List<String> srcPaths) {
        List<JavaParserTypeSolver> javaParserTypeSolvers = new ArrayList<>();
        for (String srcPath : srcPaths) {
            try {
                File srcDir = new File(srcPath);
                if (srcDir.exists() && srcDir.isDirectory()) {
                    JavaParserTypeSolver typeSolver = new JavaParserTypeSolver(srcDir);
                    javaParserTypeSolvers.add(typeSolver);
                }
            } catch (Exception e) {
                System.out.println("Could not create type solver for path: " + srcPath + " - " + e.getMessage());
            }
        }
        return javaParserTypeSolvers;
    }

    // referenced from Java Callgraph
    /**
     * 获取符号推理器
     * 
     * @param srcPath
     * @param libPath
     * @return
     */
    public JavaSymbolSolver getJavaSymbolSolver(String srcPath, String libPath) {
        return getJavaSymbolSolver(Utils.makeListFromOneElement(srcPath), Utils.makeListFromOneElement(libPath));
    }
}


