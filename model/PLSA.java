package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

public class PLSA {

	class Unit{
		double prob;
		double number;
		
		public Unit(){
			prob=0;
			number=0;
		}
		
		public Unit(double p,double n){
			prob=p;
			number=n;
		}
	}
	
	ArrayList<ArrayList<Unit>> deltaMap;
	ArrayList<HashMap<String,Unit>> tauMap;
	ArrayList<HashSet<String>> docSet;
	int numOfTopics;
	
	public PLSA(int num){
		numOfTopics=num;
		deltaMap=new ArrayList<ArrayList<Unit>>();
		tauMap=new ArrayList<HashMap<String,Unit>>();
		for(int i=0;i<numOfTopics;i++){
			tauMap.add(new HashMap<String,Unit>());
		}
		docSet=new ArrayList<HashSet<String>>();
	}
	
	public void parseTrainingFile(String fileName){
		
		try {
			BufferedReader reader=new BufferedReader(new InputStreamReader(new FileInputStream(fileName),"ISO-8859-1"));
			String line=null;
			//each time we read a line, count its words
			int doc=0;
			while((line=reader.readLine())!=null){
				while(line.isEmpty()){
					line=reader.readLine();
				}
				int count=Integer.parseInt(line);
				while(count>0){
					line=reader.readLine();
					assert(line!=null);
					count-=saveLineToSet(doc,line);
				}
				doc++;
				assert(count==0);
			}
			//close the buffered reader
			reader.close();

		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public int saveLineToSet(int doc,String line){
		String[] words=line.split(" ");
		ArrayList<String> list=new ArrayList<String>();
		for(String word:words){
			if(!word.isEmpty())
				list.add(word);
		}
		
		if(doc==docSet.size()){
			docSet.add(new HashSet<String>());
			ArrayList<Unit> deltaSubmap=new ArrayList<Unit>();
			for(int i=0;i<numOfTopics;i++){
				deltaSubmap.add(new Unit());
			}
			deltaMap.add(deltaSubmap);
		}
		
		HashSet<String> docSubmap=docSet.get(doc);
		for(String word:list){
			assert(!docSubmap.contains(word));
			docSubmap.add(word);
			for(int topic=0;topic<numOfTopics;topic++){
				HashMap<String,Unit> tauSubmap=tauMap.get(topic);
				if(!tauSubmap.containsKey(word)){
					tauSubmap.put(word,new Unit());
				}
			}
		}
		
		return list.size();
	}
	
	public void initiateParameterMaps(){
		for(HashMap<String,Unit> submap:tauMap){
			for(String word:submap.keySet()){
				submap.get(word).prob=1.0;
			}
		}
		
		Random rand=new Random();
		double mean=1.0f;
		double variance=0.02f;
		for(ArrayList<Unit> docMap:deltaMap){
			for(Unit unit:docMap){
				unit.prob=mean+rand.nextGaussian()*variance;
			}
		}
	}
	
	public void EM(){
		//E-step
		setMapNumbersToZero();
		for(int doc=0;doc<docSet.size();doc++){
			HashSet<String> wordSet=docSet.get(doc);
			for(String word:wordSet){
				double p=0;
				ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
				for(int topic=0;topic<numOfTopics;topic++){
					assert(tauMap.get(topic).containsKey(word));
					p+=deltaSubmap.get(topic).prob*tauMap.get(topic).get(word).prob;
				}
				assert(p>0);
				for(int topic=0;topic<numOfTopics;topic++){
					double q=deltaSubmap.get(topic).prob*tauMap.get(topic).get(word).prob/p;
					deltaSubmap.get(topic).number+=q;
					tauMap.get(topic).get(word).number+=q;
				}
			}
		}
		
		//M-step
		//update delta map
		for(ArrayList<Unit> deltaSubmap:deltaMap){
			double sum=0;
			for(Unit unit:deltaSubmap){
				sum+=unit.number;
			}
			for(Unit unit:deltaSubmap){
				unit.prob=unit.number/sum;
			}
		}
		//update tau map
		for(HashMap<String,Unit> tauSubmap:tauMap){
			double sum=0;
			for(String word:tauSubmap.keySet()){
				sum+=tauSubmap.get(word).number;
			}
			
			for(String word:tauSubmap.keySet()){
				Unit unit=tauSubmap.get(word);
				unit.prob=unit.number/sum;
			}
		}
	}
	
	public void setMapNumbersToZero(){
		for(ArrayList<Unit> docMap:deltaMap){
			for(Unit unit:docMap){
				unit.number=0;
			}
		}
		
		for(HashMap<String,Unit> submap:tauMap){
			for(String word:submap.keySet()){
				submap.get(word).number=0;
			}
		}
	}
	
	public double getLogLikelihood(){
		double sum=0;
		
		for(int doc=0;doc<deltaMap.size();doc++){
			ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
			HashSet<String> docSubset=docSet.get(doc);		
			for(String word:docSubset){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=deltaSubmap.get(topic).prob*tauMap.get(topic).get(word).prob;
				}
				sum+=Math.log(wordProb);
			}
		}
		
		return sum;
	}
	
	public void trainParameters(double precision){
		initiateParameterMaps();
		double prev=0;
		double cur=getLogLikelihood();
		do{
			prev=cur;
			EM();
			cur=getLogLikelihood();
		}while(Math.abs((cur-prev)/cur)<precision);
		
	}
	
	public void printDeltaProb(int doc){
		ArrayList<Unit> submap=deltaMap.get(doc);
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
	
	//public ArrayList<ArrayList<String>> getMostProbableWords(){
		
	public void smoothTauProb(double alpha){
		for(HashMap<String,Unit> tauSubmap:tauMap){
			double sum=0;
			for(String word:tauSubmap.keySet()){
				sum+=tauSubmap.get(word).number;
			}
			
			for(String word:tauSubmap.keySet()){
				Unit unit=tauSubmap.get(word);
				unit.prob=(unit.number+alpha)/(sum+alpha*tauSubmap.size());
			}
		}
	}
	
	public void getMostProbableWords(double alpha,int limit){
		smoothTauProb(alpha);
		
	}
	
	public static void main(String[] args){
		PLSA model=new PLSA(50);
		model.parseTrainingFile(args[0]);
		model.trainParameters(0.01);
		model.printDeltaProb(16);
		System.out.println("Finished");
	}
}










