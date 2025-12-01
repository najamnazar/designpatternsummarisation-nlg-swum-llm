package common.designpatternidentifier;

import java.util.*;

import org.apache.commons.collections4.MultiValuedMap;

import dps_nlg.summarygenerator.messages.DesignPatternClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternInheritClassMessage;
import dps_nlg.summarygenerator.messages.DesignPatternMethodMessage;
import dps_nlg.utils.Utils;

// Factory Method is a creational design pattern that provides an interface for 
// creating objects in a superclass, but allows subclasses to alter the type of objects that will be created.
public class FactoryPattern extends DesignPatterns {

    public FactoryPattern() {
        super("factory_method");
    }

    // Factory:
    // Has an interface - Product
    // Concrete Products implement Product
    // Creator class declares Factory method and returns Product
    // Concrete creators override factory method to return different Concrete
    // Products
    @Override
    public HashMap checkPattern(HashMap<String, HashMap> fileDetails) {

        HashMap factories = new HashMap<>();

        // Find interfaces
        for (Map.Entry<String, HashMap> fileEntry : fileDetails.entrySet()) {

            // Only an interface can be a Product
            boolean isInterfaceOrNot = false;

            // ION
            for (HashMap classDetail : Utils.getClassOrInterfaceDetails(fileEntry.getValue())) {
                if (Utils.isInterfaceOrNot(classDetail)) {
                    isInterfaceOrNot = true;
                    break;
                }
            }

            if (!isInterfaceOrNot)
                continue;
            String product = fileEntry.getKey();

            // Find concrete products using product
            // Find creator using product
            ArrayList<String> concreteProducts = new ArrayList<>();
            ArrayList<String> creators = new ArrayList<>();
            for (Map.Entry<String, HashMap> possibleConcreteProductOrCreator : fileDetails.entrySet()) {
                for (HashMap classDetail : Utils
                        .getClassOrInterfaceDetails(possibleConcreteProductOrCreator.getValue())) {

                    // Concrete Product implements Product
                    if (Utils.getImplementsFrom(classDetail).contains(product)) {
                        concreteProducts.add(possibleConcreteProductOrCreator.getKey());
                        break;
                    }

                }
                for (HashMap methodDetail : Utils.getMethodDetails(possibleConcreteProductOrCreator.getValue())) {

                    // Creator creates and returns Product
                    if (Utils.getMethodReturnType(methodDetail).equals(product)) {
                        creators.add(possibleConcreteProductOrCreator.getKey());
                        break;
                    }
                }
            }

            if (creators.size() == 0 || concreteProducts.size() == 0)
                continue;

            // Find concrete creators
            ArrayList<String> creatorsWithConcreteImplementations = new ArrayList<>();
            ArrayList<String> concreteCreators = new ArrayList<>();
            for (Map.Entry<String, HashMap> possibleConcreteCreator : fileDetails.entrySet()) {
                for (HashMap classDetail : Utils.getClassOrInterfaceDetails(possibleConcreteCreator.getValue())) {

                    // Concrete Creator implements/extends Creator
                    for (String creator : creators)
                        if (Utils.getImplementsFrom(classDetail).contains(creator)
                                || Utils.getExtendsFrom(classDetail).contains(creator)) {
                            concreteCreators.add(possibleConcreteCreator.getKey());
                            creatorsWithConcreteImplementations.add(creator);
                        }
                }
            }

            if (creatorsWithConcreteImplementations.size() == 0)
                continue;

            factories.put(creators.get(0), new HashMap());
            ((HashMap) factories.get(creators.get(0))).put("abstract_product", product);
            ((HashMap) factories.get(creators.get(0))).put("concrete_factory", creatorsWithConcreteImplementations);
            ((HashMap) factories.get(creators.get(0))).put("concrete_product", concreteProducts);
        }
        
        return createPatternResult(factories);
    }

    @Override
    public void summarise(HashMap<String, HashMap> fileDetails, HashMap designPatternDetails,
            MultiValuedMap<String, String> summary) {

        // Details of factory
        for (Map.Entry<String, HashMap> factoryDetails : ((HashMap<String, HashMap>) designPatternDetails
                .getOrDefault(patternName, new HashMap<>())).entrySet()) {
            String factory = factoryDetails.getKey();
            String abstractProduct = (String) factoryDetails.getValue().getOrDefault("abstract_product", "");
            for (String concreteProduct : (ArrayList<String>) factoryDetails.getValue().getOrDefault("concrete_product", new ArrayList<>())) {
                HashMap classDetail = (HashMap) fileDetails.getOrDefault(concreteProduct, new HashMap<>());
                HashMap parentClassDetail = (HashMap) fileDetails.getOrDefault(abstractProduct, new HashMap<>());

                DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteProduct, patternNameAsText);
                ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                ArrayList<String> overrideMethodArray = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();

                overrideMethodArray.addAll(
                        Utils.checkMethodOverride(classDetail, parentClassDetail, " method of " + abstractProduct));

                mm.setOverrideMethod(overrideMethodArray);
                mm.setMethodVerb("override");
                mmal.add(mm);

                cm.setClassName(concreteProduct);
                cm.setDesignPattern("product");
                cm.setRelatedClassName(factory);

                String concreteProductSentence = sentenceGenerator.generateSentence(cm, mmal, null);
                summary.put(concreteProduct, concreteProductSentence);
            }

            for (String concreteFactory : (ArrayList<String>) factoryDetails.getValue().getOrDefault("concrete_factory", new ArrayList<>())) {

                HashMap classDetail = (HashMap) fileDetails.getOrDefault(concreteFactory, new HashMap<>());
                HashMap parentClassDetail = (HashMap) fileDetails.getOrDefault(abstractProduct, new HashMap<>());

                DesignPatternClassMessage cm = new DesignPatternClassMessage(concreteFactory, patternNameAsText);
                ArrayList<DesignPatternMethodMessage> mmal = new ArrayList<>();
                ArrayList<String> overrideMethodArray = new ArrayList<>();
                DesignPatternMethodMessage mm = new DesignPatternMethodMessage();

                overrideMethodArray.addAll(Utils.checkMethodOverride(classDetail, parentClassDetail,
                        " method of " + factory));

                mm.setOverrideMethod(overrideMethodArray);
                mm.setMethodVerb("override");
                mmal.add(mm);

                cm.setClassName(concreteFactory);
                cm.setDesignPattern("factory");
                cm.setRelatedClassName(factory);

                String concreteFactorySentence = sentenceGenerator.generateSentence(cm, mmal, null);
                summary.put(concreteFactory, concreteFactorySentence);
            }

            String concreteProductsAsText = String.join(", ",
                    (ArrayList<String>) factoryDetails.getValue().getOrDefault("concrete_product", new ArrayList<>()));
            String concreteFactoriesAsText = String.join(", ",
                    (ArrayList<String>) factoryDetails.getValue().getOrDefault("concrete_factory", new ArrayList<>()));

            // for abstract product
            DesignPatternClassMessage apcm = new DesignPatternClassMessage(abstractProduct,
                    Utils.convertToPlainText("abstract_product"));
            apcm.setRelatedClassName(factory);

            DesignPatternInheritClassMessage apicm = new DesignPatternInheritClassMessage();
            apicm.setInheritClass(concreteProductsAsText);

            String abstractProductSentence = sentenceGenerator.generateSentence(apcm, null, apicm);
            summary.put(abstractProduct, abstractProductSentence);

            // for abstract factory
            DesignPatternClassMessage afcm = new DesignPatternClassMessage(factory, patternNameAsText);
            afcm.setRelatedClassName(abstractProduct);

            DesignPatternInheritClassMessage aficm = new DesignPatternInheritClassMessage();
            aficm.setInheritClass(concreteFactoriesAsText);

            String factoryMethodSentence = sentenceGenerator.generateSentence(afcm, null, aficm);
            summary.put(factory, factoryMethodSentence);
        }
    }
}
