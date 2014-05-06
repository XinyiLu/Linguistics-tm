package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
	HashMap<String,Unit[]> tauMap;
	ArrayList<ArrayList<String>> docSet;
	int numOfTopics;
	
	public PLSA(int num){
		numOfTopics=num;
		deltaMap=new ArrayList<ArrayList<Unit>>();
		tauMap=new HashMap<String,Unit[]>();
		docSet=new ArrayList<ArrayList<String>>();
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
			docSet.add(new ArrayList<String>());
			ArrayList<Unit> deltaSubmap=new ArrayList<Unit>();
			for(int i=0;i<numOfTopics;i++){
				deltaSubmap.add(new Unit());
			}
			deltaMap.add(deltaSubmap);
		}
		
		ArrayList<String> docSubmap=docSet.get(doc);
		for(String word:list){
			docSubmap.add(word);
			if(!tauMap.containsKey(word)){
				tauMap.put(word,new Unit[numOfTopics]);
				Unit[] unitArray=tauMap.get(word);
				for(int t=0;t<numOfTopics;t++){
					unitArray[t]=new Unit();
				}
			}
		}
		
		return list.size();
	}
	
	public void initiateParameterMaps(){
		Random rand=new Random();
		for(String word:tauMap.keySet()){
			Unit[] unitArray=tauMap.get(word);
			for(int t=0;t<numOfTopics;t++){
				unitArray[t].prob=1.0;
			}
		}
		double mean=1.0;
		double variance=0.02;
		for(ArrayList<Unit> docMap:deltaMap){
			for(Unit unit:docMap){
				unit.prob=mean+(rand.nextDouble()*2-1.0)*variance;
			}
			
		}
	}
	
	public void EM(){
		//E-step
		setMapNumbersToZero();
		for(int doc=0;doc<docSet.size();doc++){
			ArrayList<String> wordSet=docSet.get(doc);
			for(String word:wordSet){
				double p=0;
				ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
				for(int topic=0;topic<numOfTopics;topic++){
					p+=(deltaSubmap.get(topic).prob*tauMap.get(word)[topic].prob);
				}
				
				for(int topic=0;topic<numOfTopics;topic++){
					double q=(deltaSubmap.get(topic).prob*tauMap.get(word)[topic].prob)/p;
					deltaSubmap.get(topic).number+=q;
					tauMap.get(word)[topic].number+=q;
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
				unit.prob=(unit.number/sum);
			}
		}
		
		for(int t=0;t<numOfTopics;t++){
			double sum=0;
			for(String word:tauMap.keySet()){
				sum+=tauMap.get(word)[t].number;
			}
			
			for(String word:tauMap.keySet()){
				Unit unit=tauMap.get(word)[t];
				unit.prob=(unit.number/sum);
			}
		}
	}
	
	public void setMapNumbersToZero(){
		for(ArrayList<Unit> docMap:deltaMap){
			for(Unit unit:docMap){
				unit.number=0.0;
			}
		}
		
		for(String word:tauMap.keySet()){
			Unit[] tauList=tauMap.get(word);
			for(Unit unit:tauList){
				unit.number=0.0;
			}
		}
	}
	
	public double getLogLikelihood(){
		double sum=0;
		
		for(int doc=0;doc<docSet.size();doc++){
			ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
			ArrayList<String> docSubset=docSet.get(doc);
			for(String word:docSubset){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=(deltaSubmap.get(topic).prob*tauMap.get(word)[topic].prob);
				}
				sum+=Math.log(wordProb);
			}
		}
		
		
		return sum;
	}
	
	public double trainParameters(){
		initiateParameterMaps();
		double cur=0;
		int count=0;
		do{
			count++;
			EM();
			cur=getLogLikelihood();
		}while(count<50);
		return cur;
	}
	
	public void printDeltaProb(int doc){
		ArrayList<Unit> submap=deltaMap.get(doc);
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
	@SuppressWarnings("unchecked")
	public ArrayList<SmoothedUnit>[] getSmoothedTauProb(double theta){
		ArrayList<SmoothedUnit>[] smoothedMap=new ArrayList[numOfTopics];
		for(int t=0;t<numOfTopics;t++){
			smoothedMap[t]=new ArrayList<SmoothedUnit>();
		}
		
		for(String word:tauMap.keySet()){
			double sum=0;
			Unit[] unitArray=tauMap.get(word);
			for(Unit unit:unitArray){
				sum+=unit.number;
			}
			
			for(int topic=0;topic<numOfTopics;topic++){
				double tempProb=(unitArray[topic].number+theta)/(sum+theta*numOfTopics);
				smoothedMap[topic].add(new SmoothedUnit(word,tempProb));
			}
		}
		return smoothedMap;
	}
	
	public ArrayList<ArrayList<String>> getMostProbableWords(double alpha,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<SmoothedUnit>[] smoothedMap=getSmoothedTauProb(alpha);
		for(int topic=0;topic<numOfTopics;topic++){
			PriorityQueue<SmoothedUnit> heap=new PriorityQueue<SmoothedUnit>(limit);
			ArrayList<SmoothedUnit> sublist=smoothedMap[topic];
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
	
public ArrayList<ArrayList<String>> getMostProbableWordsArray(double alpha,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<SmoothedUnit>[] smoothedMap=getSmoothedTauProb(alpha);
		for(int topic=0;topic<numOfTopics;topic++){
			ArrayList<SmoothedUnit> subList=smoothedMap[topic];
			SmoothedUnit[] subArray=subList.toArray(new SmoothedUnit[subList.size()]);
			Arrays.sort(subArray);
			ArrayList<String> list=new ArrayList<String>();
			for(int i=subArray.length-1;i>=subArray.length-limit;i--){
				list.add(subArray[i].word);
			}
			result.add(list);
		}
		
		for(int topic=0;topic<numOfTopics;topic++){
			System.out.println("topic:"+topic);
			ArrayList<String> sublist=result.get(topic);
			for(String word:sublist){
				System.out.print(word+"\t");
			}
			System.out.println();
		}
		return result;
	}
	
	public void printMostProbableWords(double alpha,int limit){
		ArrayList<ArrayList<String>> list=getMostProbableWords(alpha,limit);
		for(int topic=0;topic<numOfTopics;topic++){
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
		System.out.println("The log likelihood of the data:");
		System.out.println(model.trainParameters());
		System.out.println("The probability of topics for article 17:");
		model.printDeltaProb(16);
		System.out.println("The most probable 15 words w for each topic:");
		model.printMostProbableWords(5, 15);
	}
}










