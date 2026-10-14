import {plan,validateModel,validateCalibration,validateImprovedModel,validateInferredModel,validateTeamModel} from './engine.mjs';
let model,parameters,improvedModel,inferredModel,teamModel;
self.onmessage=event=>{
  try{
    if(event.data.model){model=validateModel(event.data.model);parameters=event.data.parameters?validateCalibration(event.data.parameters):null;improvedModel=event.data.improvedModel?validateImprovedModel(event.data.improvedModel):null;inferredModel=event.data.inferredModel?validateInferredModel(event.data.inferredModel):null;teamModel=event.data.teamModel?validateTeamModel(event.data.teamModel):null;self.postMessage({ready:true});return;}
    if(!model)throw Error('Model is not loaded.');
    const started=performance.now(),result=plan(model,{...event.data,parameters,improvedModel,inferredModel,teamModel});
    self.postMessage({result,milliseconds:performance.now()-started});
  }catch(error){self.postMessage({error:error.message});}
};
