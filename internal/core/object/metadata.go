package object

func (o *Object) SetMeta(key string,value any){

	o.metadata[key]=value

}

func (o *Object) Meta(key string)(any,bool){

	v,ok:=o.metadata[key]

	return v,ok

}