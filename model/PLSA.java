package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;
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
	
	@SuppressWarnings("rawtypes")
	class SmoothedUnit implements Comparable{
		String word;
		double prob;

		public SmoothedUnit(String w,double p){
			word=w;
			prob=p;
		}
		@Override
		public int compareTo(Object arg0) {
			SmoothedUnit unit=(SmoothedUnit)arg0;
			if(unit.prob<prob){
				return 1;
			}else if(unit.prob==prob){
				return 0;
			}else{
				return -1;
			}
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
					count-=saveLineToSet(doc,line);
				}
				doc++;
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
					p+=deltaSubmap.get(topic).prob*tauMap.get(topic).get(word).prob;
				}
				
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
		double cur=0;
		int count=0;
		do{
			count++;
			prev=cur;
			EM();
			cur=getLogLikelihood();
			System.out.println(cur);
		}while(count<numOfTopics);
		
	}
	
	public void printDeltaProb(int doc){
		ArrayList<Unit> submap=deltaMap.get(doc);
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
		
	public ArrayList<ArrayList<SmoothedUnit>> getSmoothedTauProb(double alpha){
		ArrayList<ArrayList<SmoothedUnit>> smoothedMap=new ArrayList<ArrayList<SmoothedUnit>>();
		for(HashMap<String,Unit> tauSubmap:tauMap){
			double sum=0;
			ArrayList<SmoothedUnit> newList=new ArrayList<SmoothedUnit>();
			for(String word:tauSubmap.keySet()){
				sum+=tauSubmap.get(word).number;
			}
			
			for(String word:tauSubmap.keySet()){
				double prob=(tauSubmap.get(word).number+alpha)/(sum+alpha*tauSubmap.size());
				newList.add(new SmoothedUnit(word,prob));
			}
			smoothedMap.add(newList);
		}
		return smoothedMap;
	}
	
	public ArrayList<ArrayList<String>> getMostProbableWords(double alpha,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<SmoothedUnit>> smoothedMap=getSmoothedTauProb(alpha);
		for(int topic=0;topic<numOfTopics;topic++){
			PriorityQueue<SmoothedUnit> heap=new PriorityQueue<SmoothedUnit>(limit);
			ArrayList<SmoothedUnit> sublist=smoothedMap.get(topic);
			for(int i=0;i<sublist.size();i++){
				SmoothedUnit unit=sublist.get(i);
				if(heap.size()<limit){
					heap.add(unit);
					continue;
				}else if(heap.peek().prob<unit.prob){
					heap.poll();
					heap.add(unit);
				}
			}
			ArrayList<String> list=new ArrayList<String>();
			while(!heap.isEmpty()){
				list.add(0,heap.poll().word);
			}
			result.add(list);
		}
		return result;
	}
	
	public void printMostProbableWords(double alpha,int limit){
		ArrayList<ArrayList<String>> list=getMostProbableWords(alpha,limit);
		for(int topic=0;topic<numOfTopics;topic++){
			System.out.println("topic:"+topic);
			ArrayList<String> sublist=list.get(topic);
			for(String word:sublist){
				System.out.print(word+"\t");
			}
			System.out.println();
		}
	}
	
	public static void main(String[] args){
		PLSA model=new PLSA(50);
		model.parseTrainingFile(args[0]);
		model.trainParameters(0.01);
		model.printDeltaProb(16);
		model.printMostProbableWords(1, 15);
		System.out.println("Finished");
	}
}










