var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );

var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

controls = new THREE.TrackballControls( camera, renderer.domElement );
                controls.minDistance = 2;
                controls.maxDistance = 200;

light = new THREE.DirectionalLight( 0xffffff );
light.position.set( 0, 0, 100 );
scene.add( light );

scene.add( new THREE.AmbientLight( 0x606060 ) );
scene.add( new THREE.AxisHelper(1) );

var geometry = new THREE.SphereGeometry( 1, 32, 32 );
var material = new THREE.MeshPhongMaterial( { shininess: 70 } );
var loader = new THREE.TextureLoader();
material.map = loader.load('fiducial_texture.jpg');
var sphere = new THREE.Mesh( geometry, material );
var baseObj = sphere;


var objects = {
};

camera.position.z = 20;

var updateScene = function(newObject) {
    var id = newObject["id"]
	if (typeof(objects[id]) === "undefined") {
	    objects[id] = baseObj.clone()
	    scene.add(objects[id])
	}
	objects[id].position.fromArray(newObject["position"]);
	objects[id].quaternion.fromArray(newObject["orientation"]);
}



var render = function () {
	requestAnimationFrame( render );
	camera.lookAt( scene.position );
	controls.update();
	renderer.render(scene, camera);
};


if ("WebSocket" in window)
    {
       var uri = "ws://localhost:8888/ws"
       var ws = new WebSocket(uri);

       ws.onopen = function()
       {
          // Web Socket is connected, send data using send()
          // ws.send("Message to send");
          //alert("Connection open");
       };

       ws.onmessage = function (evt)
       {
          var received_msg = evt.data;
          var myItem = JSON.parse(received_msg);
          updateScene(myItem);
       };

       ws.onclose = function()
       {
          // websocket is closed.
          //alert("Connection Closed...");
       };

       ws.onerror = function()
       {
          // websocket is closed.
          alert("No visualiser server detected.\n\nPlease ensure you have a websocket server running and accepting connections at " + uri + ".");
       };

    }
    else
    {
       alert("WebSocket NOT supported by your Browser!");
    }

render();